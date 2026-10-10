# =============================================================================
# lowrank_glm.jl — optimization-based (non-Bayesian) low-rank GLM factor models
#
#   * Poisson PCA            (counts)          fit_poisson_pca
#   * Logistic / binary PCA  (0/1 data)        fit_logistic_pca
#   * Negative-binomial PCA  (NB2 counts)       fit_nb_pca
#   * ZINB-WaVE-style PCA    (zero-inflated NB) fit_zinb_pca
#
# Convention: rows = cells / samples (n), columns = genes / features (p).
#   η_ij = o_i + b_j + x_i' y_j        (natural parameter, Θ = X*Y')
#   X (n×k) scores, Y (p×k) loadings, b (p) per-column intercept,
#   o (n) fixed row offset (e.g. log library size, see `log_libsize`).
#   Loss = Σ_{observed ij} NLL(a_ij; η_ij) + λ/2 (‖X‖² + ‖Y‖²)   (intercepts unpenalized)
#
# Solvers (keyword `solver`):
#   :newton (default) alternating block minimisation, described below;
#   :adam   row-mini-batch Adam with analytic gradients (BLAS gemms, O(npk) per
#           epoch instead of O(npk²) per Newton sweep), plateau step-halving;
#   :hybrid Adam (warm start) followed by at most `polish_iters` Newton sweeps.
#
# Fitting (:newton): alternating block minimisation. Given Y every row x_i is a small
# GLM (k params), given X every row of [Y b] is a small GLM (k+1 params).
# Each block takes `inner` damped Newton / Fisher-scoring (IRLS) steps with an
# Armijo backtracking line search, threaded with Threads.@spawn, preallocated
# per-thread buffers, in-place updates. ZINB uses the same engine: the Newton
# curvature comes from the EM surrogate (posterior zero-inflation weights z_ij),
# which is tangent to the true ZINB likelihood, so the line search on the true
# likelihood makes every block step monotone. Per-gene dispersion θ_j = exp(ζ_j)
# is updated by safeguarded 1-D Newton on ζ_j (Float64 internally).
#
# Required packages:
#   stdlib:     LinearAlgebra, Random, Statistics, SparseArrays
#   registered: LogExpFunctions (log1pexp, logistic, logaddexp, logit)
#               SpecialFunctions (loggamma, digamma, trigamma)
# Run with threads, e.g.  julia -t auto --project=. src/lowrank_glm.jl  (from the repo root)
# =============================================================================

using LinearAlgebra, Random, Statistics, SparseArrays
using LogExpFunctions: log1pexp, logistic, logaddexp, logit
using SpecialFunctions: loggamma, digamma, trigamma

struct PoissonFamily end
struct BernoulliFamily end
struct NBFamily end          # NB2, per-gene dispersion; aux[j] = (ζ_j, θ_j), θ = exp(ζ)

# constant "vector" (every index returns the same value): per-column aux / no aux
struct _Cst{V}; v::V; end
@inline Base.getindex(c::_Cst, i) = c.v
const _NOAUX = _Cst(nothing)

# ----------------------------------------------------------------------------
# Per-entry losses. `_entry` returns (reduced NLL, dNLL/dη, Fisher weight);
# "reduced" drops terms that do not depend on η (used in line searches).
# ----------------------------------------------------------------------------
@inline function _entry(::PoissonFamily, a::T, η::T, aux = nothing) where {T}
    μ = exp(η)
    return (μ - a * η, μ - a, μ)
end
@inline _nll(::PoissonFamily, a::T, η::T, aux = nothing) where {T} = exp(η) - a * η
@inline _nll_full(::PoissonFamily, a, η, aux = nothing) =
    (a64 = Float64(a); e = Float64(η); exp(e) - a64 * e + (a64 > 0 ? loggamma(a64 + 1) : 0.0))

@inline function _entry(::BernoulliFamily, a::T, η::T, aux = nothing) where {T}
    pr = logistic(η)
    return (log1pexp(η) - a * η, pr - a, max(pr * (one(T) - pr), eps(T)))
end
@inline _nll(::BernoulliFamily, a::T, η::T, aux = nothing) where {T} = log1pexp(η) - a * η
@inline _nll_full(::BernoulliFamily, a, η, aux = nothing) = (e = Float64(η); log1pexp(e) - Float64(a) * e)

# NB2: with u = η - ζ, L = log(1+μ/θ) = log1pexp(u), r = μ/(μ+θ) = logistic(u):
#   NLL = θL - a(u - L) - [lgamma(a+θ) - lgamma(θ) - lgamma(a+1)];  dNLL/dη = θr - a(1-r);
#   Fisher weight θr = μθ/(μ+θ) = μ/(1+μ/θ)
@inline function _entry(::NBFamily, a::T, η::T, aux) where {T}
    ζ, θ = aux; u = η - ζ; L = log1pexp(u); r = logistic(u)
    return (θ * L - a * (u - L), θ * r - a * (one(T) - r), max(θ * r, eps(T)))
end
@inline _nll(::NBFamily, a::T, η::T, aux) where {T} = (u = η - aux[1]; L = log1pexp(u); aux[2] * L - a * (u - L))

# ZINB: μ = exp(ηm), π = logistic(ηp), θ = exp(ζ).  With u = ηm - ζ:
#   L = log(1 + μ/θ) = log1pexp(u),  r = μ/(μ+θ) = logistic(u)
#   a = 0:  -log[π + (1-π)(1+μ/θ)^(-θ)] = -logaddexp(log π, log(1-π) - θL)
#   a > 0:  log1pexp(ηp) + θL - a(u - L) - [lgamma(a+θ) - lgamma(θ) - lgamma(a+1)]
# Returns (reduced NLL, dℓ/dηm, dℓ/dηp, wm, wp); wm, wp = EM-surrogate Fisher weights.
@inline function _zinb_entry(a::T, ηm::T, ηp::T, ζ::T, θ::T) where {T}
    u = ηm - ζ
    L = log1pexp(u); r = logistic(u); pz = logistic(ηp)
    wp = max(pz * (one(T) - pz), eps(T))
    lq = log1pexp(ηp)                        # -log(1-π);  log π = ηp - lq
    if iszero(a)
        lp0 = logaddexp(ηp - lq, -lq - θ * L)
        z = exp(ηp - lq - lp0)               # posterior P(structural zero | a = 0)
        gm = (one(T) - z) * θ * r
        return (-lp0, gm, pz - z, gm, wp)
    else
        return (lq + θ * L - a * (u - L), θ * r - a * (one(T) - r), pz, θ * r, wp)
    end
end
@inline function _zinb_nll(a::T, ηm::T, ηp::T, ζ::T, θ::T) where {T}   # reduced
    u = ηm - ζ; L = log1pexp(u); lq = log1pexp(ηp)
    iszero(a) ? -logaddexp(ηp - lq, -lq - θ * L) : lq + θ * L - a * (u - L)
end
# full NLL in Float64 (includes lgamma terms; `withconst` adds lgamma(a+1))
@inline function _zinb_nll_full(a, ηm, ηp, ζ; withconst::Bool = true)
    a, ηm, ηp, ζ = Float64(a), Float64(ηm), Float64(ηp), Float64(ζ)
    θ = exp(ζ)
    v = _zinb_nll(a, ηm, ηp, ζ, θ)
    a > 0 && (v -= loggamma(a + θ) - loggamma(θ) - (withconst ? loggamma(a + 1) : 0.0))
    return v
end
@inline _nll_full(::NBFamily, a, η, aux) = _zinb_nll_full(a, η, -Inf, aux[1])   # NB = ZINB with π ≡ 0
_nbaux(ζ::AbstractVector{T}) where {T} = [(ζ[j], exp(ζ[j])) for j in eachindex(ζ)]

# derivatives wrt ζ = log θ (Float64): (NLL without lgamma(a+1), d/dζ, curvature)
@inline function _zinb_disp_entry(a::Float64, ηm::Float64, ηp::Float64, ζ::Float64, θ::Float64)
    u = ηm - ζ; L = log1pexp(u); r = logistic(u)
    lq = log1pexp(ηp)
    if a == 0
        lp0 = logaddexp(ηp - lq, -lq - θ * L)
        z = exp(ηp - lq - lp0)
        return (-lp0, -(1 - z) * θ * (r - L), -(1 - z) * θ * (r - L + r^2))
    else
        nll = lq + θ * L - a * (u - L) - loggamma(a + θ) + loggamma(θ)
        F1 = -(digamma(a + θ) - digamma(θ) - L + r - a * (1 - r) / θ)
        F2 = -(trigamma(a + θ) - trigamma(θ) + r^2 / θ + a * (1 - r)^2 / θ^2)
        return (nll, θ * F1, θ * F1 + θ^2 * F2)
    end
end

# ----------------------------------------------------------------------------
# Data container (dense or sparse, optional observation mask)
# ----------------------------------------------------------------------------
struct _Data{T,MA<:AbstractMatrix{T},MK}
    A::MA      # n×p
    At::MA     # p×n transpose (sparse: fast row access); dense: same object as A
    mask::MK   # Nothing (all observed) or BitMatrix (true = observed)
end

function _prepare(A::SparseMatrixCSC{<:Any,Ti}, mask, ::Type{T}) where {T,Ti}
    S = SparseMatrixCSC{T,Ti}(A)                   # keeps Int32 indices if given (half the memory)
    M = mask === nothing ? nothing : BitMatrix(mask)
    return _Data(S, sparse(transpose(S)), M)
end
function _prepare(A::AbstractMatrix, mask, ::Type{T}) where {T}
    if Missing <: eltype(A)
        obs = BitMatrix(.!ismissing.(A))
        Ad = Matrix{T}(coalesce.(A, zero(T)))
        M = mask === nothing ? obs : obs .& BitMatrix(mask)
    else
        Ad = Matrix{T}(A)
        M = mask === nothing ? nothing : BitMatrix(mask)
    end
    M === nothing || (Ad[.!M] .= zero(T))
    return _Data(Ad, Ad, M)
end

_maskvec!(w, ::Nothing, i, ::Val) = fill!(w, one(eltype(w)))
_maskvec!(w, M::BitMatrix, i, ::Val{:row}) = (@inbounds for j in eachindex(w); w[j] = M[i, j]; end; w)
_maskvec!(w, M::BitMatrix, j, ::Val{:col}) = (@inbounds for i in eachindex(w); w[i] = M[i, j]; end; w)

function _getcol!(a, w, D::_Data{T,Matrix{T}}, j) where {T}
    @inbounds for i in eachindex(a); a[i] = D.A[i, j]; end
    _maskvec!(w, D.mask, j, Val(:col))
end
function _getrow!(a, w, D::_Data{T,Matrix{T}}, i) where {T}
    @inbounds for j in eachindex(a); a[j] = D.A[i, j]; end
    _maskvec!(w, D.mask, i, Val(:row))
end
function _spcol!(a, S::SparseMatrixCSC, j)
    fill!(a, zero(eltype(a))); rv = rowvals(S); nz = nonzeros(S)
    @inbounds for idx in nzrange(S, j); a[rv[idx]] = nz[idx]; end
    a
end
_getcol!(a, w, D::_Data{T,<:SparseMatrixCSC}, j) where {T} = (_spcol!(a, D.A, j); _maskvec!(w, D.mask, j, Val(:col)))
_getrow!(a, w, D::_Data{T,<:SparseMatrixCSC}, i) where {T} = (_spcol!(a, D.At, i); _maskvec!(w, D.mask, i, Val(:row)))

# ----------------------------------------------------------------------------
# Small numerical kernels (allocation-free)
# ----------------------------------------------------------------------------
struct _Buf{T}
    a::Vector{T}; w::Vector{T}; c::Vector{T}; c2::Vector{T}
    η::Vector{T}; η2::Vector{T}; ηt::Vector{T}; ηt2::Vector{T}
    g::Vector{T}; g2::Vector{T}; h::Vector{T}; h2::Vector{T}
    H::Matrix{T}; H2::Matrix{T}; gβ::Vector{T}; gβ2::Vector{T}
    β::Vector{T}; β2::Vector{T}; βt::Vector{T}; βt2::Vector{T}; Δ::Vector{T}; Δ2::Vector{T}
    acc::Base.RefValue{Float64}
end
_Buf(::Type{T}, m, q) where {T} = _Buf{T}(
    (zeros(T, m) for _ in 1:12)..., zeros(T, q, q), zeros(T, q, q),
    (zeros(T, q) for _ in 1:8)..., Ref(0.0))

# run f(i, buf) for i in 1:N over contiguous chunks, one buffer per task
function _tforeach(f::F, N::Int, bufs) where {F}
    nb = length(bufs)
    if nb == 1 || N < 2nb
        B1 = bufs[1]
        for i in 1:N; f(i, B1); end
        return nothing
    end
    cs = cld(N, nb)
    @sync for c in 1:nb
        lo = (c - 1) * cs + 1; hi = min(c * cs, N)
        lo > hi && continue
        Threads.@spawn begin
            local Bc = bufs[c]
            for i in lo:hi; f(i, Bc); end
        end
    end
    return nothing
end

# η = c + D[:, 1:q] * β
@inline function _eta!(η, D, β, c, q)
    m = length(η)
    @inbounds @simd for t in 1:m; η[t] = c[t]; end
    @inbounds for r in 1:q
        br = β[r]
        @simd for t in 1:m; η[t] = muladd(D[t, r], br, η[t]); end
    end
    η
end
# H = D' diag(h) D,  gβ = D' g   (columns 1:q).  (A BLAS variant was benchmarked
# and gave no gain up to k = 50: the per-entry exp/log work dominates.)
@inline function _hess_grad!(H, gβ, D, h, g, q)
    m = length(h); T = eltype(H)
    @inbounds for r in 1:q
        s = zero(T)
        @simd for t in 1:m; s = muladd(D[t, r], g[t], s); end
        gβ[r] = s
        for r2 in r:q
            v = zero(T)
            @simd for t in 1:m; v = muladd(h[t] * D[t, r], D[t, r2], v); end
            H[r, r2] = v; H[r2, r] = v
        end
    end
end
# in-place Cholesky solve of H x = rhs (x holds rhs on entry); false if not PD
@inline function _cholsolve!(H, x, q)
    @inbounds begin
        for j in 1:q
            s = H[j, j]
            for l in 1:j-1; s -= H[j, l]^2; end
            s > 0 || return false
            d = sqrt(s); H[j, j] = d
            for i in j+1:q
                t = H[i, j]
                for l in 1:j-1; t -= H[i, l] * H[j, l]; end
                H[i, j] = t / d
            end
        end
        for i in 1:q
            t = x[i]; for l in 1:i-1; t -= H[i, l] * x[l]; end; x[i] = t / H[i, i]
        end
        for i in q:-1:1
            t = x[i]; for l in i+1:q; t -= H[l, i] * x[l]; end; x[i] = t / H[i, i]
        end
    end
    return true
end
# solve (H + diag(λv)) Δ = gβ + λv.*β ; returns directional decrease gβ'Δ
@inline function _newton_dir!(Δ, H, gβ, β, λv, q)
    dmax = zero(eltype(H))
    @inbounds for r in 1:q
        H[r, r] += λv[r]; gβ[r] += λv[r] * β[r]; Δ[r] = gβ[r]; dmax = max(dmax, H[r, r])
    end
    if !_cholsolve!(H, Δ, q)                       # fallback: scaled gradient step
        @inbounds for r in 1:q; Δ[r] = gβ[r] / (dmax + one(dmax)); end
    end
    dec = 0.0
    @inbounds for r in 1:q; dec += Float64(gβ[r]) * Float64(Δ[r]); end
    return dec
end
@inline _pen(β, λv, q) = (s = 0.0; @inbounds for r in 1:q; s += 0.5 * Float64(λv[r]) * Float64(β[r])^2; end; s)

function _loss_derivs!(g, h, fam, a, w, η, aux)
    f = 0.0; T = eltype(η)
    @inbounds for t in eachindex(a)
        if iszero(w[t]); g[t] = zero(T); h[t] = zero(T); continue; end
        ℓ, gt, ht = _entry(fam, a[t], η[t], aux[t])
        f += w[t] * ℓ; g[t] = w[t] * gt; h[t] = w[t] * ht
    end
    return f
end
function _loss(fam, a, w, η, aux)
    f = 0.0
    @inbounds for t in eachindex(a)
        iszero(w[t]) || (f += w[t] * _nll(fam, a[t], η[t], aux[t]))
    end
    return f
end

# `nsteps` damped Newton/IRLS steps for one block: η = c + D[:,1:q]β; aux[t] = per-entry
# family parameter (NB: (ζ, θ) of the entry's gene), `_NOAUX` for Poisson / Bernoulli
function _block_newton!(β, fam, D, c, a, w, q, λv, nsteps, B, aux = _NOAUX)
    η, ηt, βt, Δ = B.η, B.ηt, B.βt, B.Δ
    T = eltype(β)
    _eta!(η, D, β, c, q)
    f0 = _loss_derivs!(B.g, B.h, fam, a, w, η, aux) + _pen(β, λv, q)
    for step in 1:nsteps
        _hess_grad!(B.H, B.gβ, D, B.h, B.g, q)
        dec = _newton_dir!(Δ, B.H, B.gβ, β, λv, q)
        (dec > 0 && isfinite(dec)) || break
        s = one(T); ok = false
        for _ in 1:12
            @inbounds for r in 1:q; βt[r] = β[r] - s * Δ[r]; end
            _eta!(ηt, D, βt, c, q)
            f1 = _loss(fam, a, w, ηt, aux) + _pen(βt, λv, q)
            if f1 <= f0 - 1e-4 * s * dec
                @inbounds for r in 1:q; β[r] = βt[r]; end
                copyto!(η, ηt); f0 = f1; ok = true; break
            end
            s /= 2
        end
        ok || break
        step < nsteps && (f0 = _loss_derivs!(B.g, B.h, fam, a, w, η, aux) + _pen(β, λv, q))
    end
    return f0
end

# ----------------------------------------------------------------------------
# Initialisation: randomized SVD of a (possibly sparse) matrix with implicit
# column centring  R = Z - 1 v'
# ----------------------------------------------------------------------------
function _rsvd_centered(Z, v::Vector{T}, k; rng = Random.default_rng(), power = 2, os = 10) where {T}
    n, p = size(Z); l = min(k + os, n, p)
    mulR(M) = Z * M .- ones(T, n) * (v' * M)
    mulRt(M) = Z' * M .- v * sum(M; dims = 1)
    Q = Matrix(qr(mulR(randn(rng, T, p, l))).Q)
    for _ in 1:power
        Q = Matrix(qr(mulRt(Q)).Q)
        Q = Matrix(qr(mulR(Q)).Q)
    end
    F = svd(Matrix(mulRt(Q)'))
    return Q * F.U[:, 1:k], F.S[1:k], F.V[:, 1:k]
end

# SVD start: counts -> centred log1p(size-normalised counts), binary -> centred A.
function _init_svd(data::_Data{T}, o, k, binary::Bool, rng) where {T}
    A = data.A; M = data.mask; n, p = size(A)
    s = exp.(o .- mean(o))
    if binary
        Z = copy(A)
    elseif A isa SparseMatrixCSC
        Z = copy(A); nz = nonzeros(Z); rv = rowvals(Z)
        @inbounds for idx in eachindex(nz); nz[idx] = log1p(nz[idx] / s[rv[idx]]); end
    else
        Z = log1p.(A ./ s)
    end
    a = zeros(T, n); w = zeros(T, n); eo = exp.(o)
    v = zeros(T, p); b = zeros(T, p); z = zeros(T, n)
    for j in 1:p
        _getcol!(a, w, data, j)
        Z isa SparseMatrixCSC ? _spcol!(z, Z, j) : (z .= view(Z, :, j))
        nw = max(sum(w), one(T))
        v[j] = dot(w, z) / nw
        b[j] = binary ? logit(clamp(dot(w, a) / nw, T(0.02), T(0.98))) :
                        log((dot(w, a) + T(0.5)) / max(dot(w, eo), eps(T)))
    end
    if M !== nothing && Z isa Matrix              # missing -> column mean (0 after centring)
        for j in 1:p, i in 1:n; M[i, j] || (Z[i, j] = v[j]); end
    end
    U, S, V = _rsvd_centered(Z, v, k; rng)
    sc = binary ? T(2) : one(T)                  # logit ≈ 4(p - 1/2) near 1/2
    return sc .* U .* sqrt.(S)', sc .* V .* sqrt.(S)', b
end

"""
    orthogonalize(X, Y, b) -> (X, Y, b, d)

Re-express the fit `Θ = X*Y' .+ b'` in canonical form: centre the columns of X
(absorbing the means into the intercept `b`), then SVD the low-rank part via QR
of X and Y so that `Y` has orthonormal columns, `X = U*Diagonal(d)` has
orthogonal columns ordered by decreasing singular value `d`, and signs are
fixed (largest-|.| loading in each column positive). `Θ` is unchanged.
For ZINB pass the stacked loadings `[Ymu; Ypi]` and intercepts `[bmu; bpi]`.
"""
function orthogonalize(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    x̄ = vec(mean(X; dims = 1))
    b2 = b .+ Y * x̄
    Fx = qr(X .- x̄'); Fy = qr(Y)
    F = svd(Fx.R * Fy.R')
    Xn = Matrix(Fx.Q) * F.U .* F.S'
    Yn = Matrix(Fy.Q) * F.V
    for r in axes(Yn, 2)
        i = argmax(abs.(view(Yn, :, r)))
        if Yn[i, r] < 0; Yn[:, r] .*= -1; Xn[:, r] .*= -1; end
    end
    return Xn, Yn, b2, F.S
end

# ----------------------------------------------------------------------------
# Poisson / logistic / NB engine  (aux: per-gene vector of (ζ_j, θ_j) for NB, else _NOAUX)
# ----------------------------------------------------------------------------
function _sweep_rows!(fam, data, Xa, Ya, o, λv, inner, bufs, aux = _NOAUX)
    n = size(Xa, 1); k = size(Xa, 2) - 1; p = size(Ya, 1)
    _tforeach(n, bufs) do i, B
        _getrow!(B.a, B.w, data, i)
        oi = o[i]
        @inbounds for j in 1:p; B.c[j] = Ya[j, k+1] + oi; end
        @inbounds for r in 1:k; B.β[r] = Xa[i, r]; end
        _block_newton!(B.β, fam, Ya, B.c, B.a, B.w, k, λv, inner, B, aux)
        @inbounds for r in 1:k; Xa[i, r] = B.β[r]; end
    end
end
# column sweep; for NB (ζ !== nothing) also `disp_steps` Newton steps on ζ_j = log θ_j
function _sweep_cols!(fam, data, Xa, Ya, o, λv, inner, bufs, aux = _NOAUX, ζ = nothing, λd = 0.0, ζbar = 0.0,
                      disp_steps = 0)
    p = size(Ya, 1); q = size(Ya, 2)
    _tforeach(p, bufs) do j, B
        _getcol!(B.a, B.w, data, j)
        @inbounds for r in 1:q; B.β[r] = Ya[j, r]; end
        _block_newton!(B.β, fam, Xa, o, B.a, B.w, q, λv, inner, B, _Cst(aux[j]))
        @inbounds for r in 1:q; Ya[j, r] = B.β[r]; end
        if ζ !== nothing                           # B.η holds the final η of column j
            ζ[j] = eltype(ζ)(_zinb_disp(ζ[j], B.η, _Cst(-Inf), B.a, B.w, Float64(λd), ζbar, disp_steps))
        end
    end
end
function _objective(fam, data, Xa, Ya, o, λ, bufs, aux = _NOAUX)
    p = size(Ya, 1); q = size(Ya, 2); k = q - 1
    foreach(B -> (B.acc[] = 0.0), bufs)
    _tforeach(p, bufs) do j, B
        _getcol!(B.a, B.w, data, j)
        @inbounds for r in 1:q; B.β[r] = Ya[j, r]; end
        _eta!(B.η, Xa, B.β, o, q)
        f = 0.0; auxj = aux[j]
        @inbounds for t in eachindex(B.a)
            iszero(B.w[t]) || (f += B.w[t] * _nll_full(fam, B.a[t], B.η[t], auxj))
        end
        B.acc[] += f
    end
    return sum(B -> B.acc[], bufs) +
           0.5 * λ * (sum(abs2, view(Xa, :, 1:k)) + sum(abs2, view(Ya, :, 1:k)))
end

# objective penalty on the NB log-dispersions
_disp_pen(::Nothing, λd) = 0.0
_disp_pen(ζ, λd) = (m = mean(Float64, ζ); 0.5 * λd * sum(z -> (Float64(z) - m)^2, ζ))

function _als!(fam, data, Xa, Ya, o, λ::T, maxiter, tol, inner, verbose, time_limit = Inf,
               ζ = nothing, λd = 0.0, disp_steps = 2) where {T}
    n, q = size(Xa); k = q - 1; p = size(Ya, 1); t0 = time()
    nt = Threads.nthreads()
    bufX = [_Buf(T, p, q) for _ in 1:nt]; bufY = [_Buf(T, n, q) for _ in 1:nt]
    λX = fill(λ, q); λY = [fill(λ, k); zero(T)]
    trace = Float64[]; conv = false
    for it in 1:maxiter
        aux = ζ === nothing ? _NOAUX : _nbaux(ζ)
        _sweep_rows!(fam, data, Xa, Ya, o, λX, inner, bufX, aux)
        ζbar = ζ === nothing ? 0.0 : mean(Float64, ζ)
        _sweep_cols!(fam, data, Xa, Ya, o, λY, inner, bufY, aux, ζ, λd, ζbar, disp_steps)
        aux = ζ === nothing ? _NOAUX : _nbaux(ζ)
        f = _objective(fam, data, Xa, Ya, o, λ, bufY, aux) + _disp_pen(ζ, λd)
        push!(trace, f)
        verbose && println("iter $it  objective $(round(f; sigdigits = 10))")
        if it > 1 && abs(trace[end-1] - f) <= tol * (abs(f) + 1)
            conv = true; break
        end
        time() - t0 > time_limit && break
    end
    return trace, conv
end

function _fit_single(fam, A, k::Integer; offset = nothing, mask = nothing, T::Type{<:AbstractFloat} = Float32,
                     λ = 1.0, maxiter::Integer = 200, tol = 1e-6, inner::Integer = 1, init::Symbol = :svd,
                     nstarts::Integer = 1, rng = Random.default_rng(), verbose::Bool = false,
                     solver::Symbol = :newton, lr = 0.02, batchsize = nothing, epochs::Integer = 200,
                     adam_epochs::Integer = epochs, polish_iters::Integer = 20, adam_tol = 1e-5,
                     time_limit = Inf, λdisp = 10.0, disp_steps::Integer = 2)
    solver in (:newton, :adam, :hybrid) || throw(ArgumentError("solver ∈ (:newton, :adam, :hybrid)"))
    data = _prepare(A, mask, T); n, p = size(data.A)
    isnb = fam isa NBFamily
    bs = batchsize === nothing ? clamp(n ÷ 16, 64, 1024) : Int(batchsize)
    0 < k < min(n, p) || throw(ArgumentError("need 0 < k < min(n, p)"))
    o = offset === nothing ? zeros(T, n) : T.(collect(offset))
    length(o) == n || throw(DimensionMismatch("offset must have length n = $n"))
    X0, Y0, b0 = _init_svd(data, o, k, fam isa BernoulliFamily, rng)
    ζ0 = isnb ? _init_disp(data) : nothing
    best = nothing
    for s in 1:nstarts
        if s == 1 && init === :svd
            X, Y = copy(X0), copy(Y0)
        else
            X, Y = T(0.1) .* randn(rng, T, n, k), T(0.1) .* randn(rng, T, p, k)
        end
        Xa = hcat(X, ones(T, n)); Ya = hcat(Y, copy(b0)); ζ = isnb ? copy(ζ0) : nothing
        t0 = time()
        if solver === :newton
            trace, conv = _als!(fam, data, Xa, Ya, o, T(λ), maxiter, tol, inner, verbose, time_limit, ζ, λdisp, disp_steps)
        else
            ne = solver === :adam ? epochs : adam_epochs
            trace, conv = _adam_single!(fam, data, Xa, Ya, o, T(λ), lr, bs, ne, adam_tol, rng, verbose, time_limit,
                                        ζ, λdisp)
            if solver === :hybrid
                tr2, conv = _als!(fam, data, Xa, Ya, o, T(λ), min(maxiter, polish_iters), tol, inner, verbose,
                                  time_limit - (time() - t0), ζ, λdisp, disp_steps)
                trace = vcat(trace, tr2)
            end
        end
        if best === nothing || trace[end] < best[3][end]
            best = (Xa, Ya, trace, conv, ζ)
        end
    end
    Xa, Ya, trace, conv, ζ = best
    X, Y, b, d = orthogonalize(Xa[:, 1:k], Ya[:, 1:k], Ya[:, k+1])
    isnb && return (X = X, Y = Y, intercept = b, dispersion = exp.(ζ), offset = o, d = d, loss = trace,
                    converged = conv, iters = length(trace), family = :nb)
    return (X = X, Y = Y, intercept = b, offset = o, d = d, loss = trace, converged = conv,
            iters = length(trace), family = fam isa PoissonFamily ? :poisson : :bernoulli)
end

"""
    fit_poisson_pca(A, k; offset=nothing, mask=nothing, T=Float32, λ=1, maxiter=200,
                    tol=1e-6, inner=1, init=:svd, nstarts=1, rng, verbose=false)

Poisson PCA / GLM-PCA: `a_ij ~ Poisson(exp(o_i + b_j + x_i'y_j))`.
`A` (n×p counts) may be a dense matrix, a `SparseMatrixCSC` (kept sparse: no
dense n×p copy, but the Poisson loss still costs O(npk) per sweep because
`Σ exp(η)` runs over all entries), or contain `missing`. `mask` (n×p Bool,
true = observed) is AND-ed with the non-missing pattern; only observed entries
enter the loss. `offset`: fixed length-n row offset (e.g. `log_libsize(A)`).
`λ`: ridge on X and Y. `inner`: Newton steps per block per sweep. `nstarts>1`
adds random restarts (keeps the lowest objective). Convergence: relative
objective change < `tol`. `time_limit` (seconds) stops early (converged=false).

Solver (`solver`):
* `:newton` (default) alternating per-row damped Newton/IRLS (`maxiter`, `tol`, `inner`).
* `:adam` row-mini-batch Adam: each step samples `batchsize` rows (default
  `clamp(n÷16, 64, 1024)`), takes analytic gradients with BLAS gemms, updates
  those rows of X and all of [Y b] (Y-gradient scaled by n/batch). Step size
  `lr` is halved whenever an epoch fails to lower the full objective by
  `adam_tol` (relative); stops after 4 halvings (converged) or `epochs`.
  Sparse input is densified one batch at a time only.
* `:hybrid` Adam (at most `adam_epochs` epochs, default = `epochs`) as warm start,
  then at most `polish_iters` (20) Newton sweeps; `converged` refers to the Newton
  stage (relative change < `tol` within the polish budget).

Returns a NamedTuple `(X, Y, intercept, offset, d, loss, converged, iters, family)`
with X = U*Diagonal(d) (n×k), Y orthonormal (p×k); `loss` is the per-sweep
penalised NLL trace (full NLL, incl. log a!).
"""
fit_poisson_pca(A, k::Integer; kw...) = _fit_single(PoissonFamily(), A, k; kw...)

"""
    fit_logistic_pca(A, k; kwargs...)

Logistic (binary) PCA: `a_ij ~ Bernoulli(logistic(o_i + b_j + x_i'y_j))`,
`a_ij ∈ {0,1}` (fractions in [0,1] also work as soft labels). Same keywords and
return value as [`fit_poisson_pca`](@ref) (offset defaults to 0).
"""
fit_logistic_pca(A, k::Integer; kw...) = _fit_single(BernoulliFamily(), A, k; kw...)

"""
    fit_nb_pca(A, k; offset=nothing, mask=nothing, T=Float32, λ=1, λdisp=10, maxiter=300, tol=1e-6,
               inner=1, disp_steps=2, solver=:newton, kwargs...)

Negative-binomial (NB2) PCA without zero inflation:
`a_ij ~ NB(μ_ij, θ_j)`, `log μ_ij = o_i + b_j + x_i'y_j`, `Var = μ + μ²/θ_j`.
Per-gene dispersion θ_j = exp(ζ_j) (log-parameterised, clamped to [1e-3, 1e5]),
shrunk towards the mean log-dispersion by `λdisp/2 Σ (ζ_j - ζ̄)²` (as in
[`fit_zinb_pca`](@ref)). Newton: Fisher scoring with NB2 weights μ/(1+μ/θ),
plus `disp_steps` safeguarded Newton steps on each ζ_j per sweep; Adam: ζ is an
extra Adam parameter. Inputs (dense / sparse / missing / mask / offset), `solver`
(:newton / :adam / :hybrid) and all other keywords as in [`fit_poisson_pca`](@ref).

Returns `(X, Y, intercept, dispersion, offset, d, loss, converged, iters, family = :nb)`.
"""
fit_nb_pca(A, k::Integer; maxiter::Integer = 300, kw...) = _fit_single(NBFamily(), A, k; maxiter, kw...)

# ----------------------------------------------------------------------------
# ZINB engine (shared scores X; μ and π have their own loadings/intercepts)
# ----------------------------------------------------------------------------
# row i of X (shared by both predictors); θ, ζ are per-gene vectors
function _zinb_row!(β, Dm, Dp, cm, cp, a, w, ζ, θ, k, λv, nsteps, zi::Bool, B)
    T = eltype(β); η, η2, ηt, ηt2, g, g2, h, h2 = B.η, B.η2, B.ηt, B.ηt2, B.g, B.g2, B.h, B.h2
    function lossd!()
        f = 0.0
        @inbounds for t in eachindex(a)
            if iszero(w[t]); g[t] = h[t] = g2[t] = h2[t] = zero(T); continue; end
            ℓ, gm, gp, wm, wp = _zinb_entry(a[t], η[t], η2[t], ζ[t], θ[t])
            f += w[t] * ℓ; g[t] = w[t] * gm; h[t] = w[t] * wm; g2[t] = w[t] * gp; h2[t] = w[t] * wp
        end
        f
    end
    _eta!(η, Dm, β, cm, k); zi ? _eta!(η2, Dp, β, cp, k) : copyto!(η2, cp)
    f0 = lossd!() + _pen(β, λv, k)
    for step in 1:nsteps
        _hess_grad!(B.H, B.gβ, Dm, h, g, k)
        if zi
            _hess_grad!(B.H2, B.gβ2, Dp, h2, g2, k)
            @inbounds for r in 1:k, r2 in 1:k; B.H[r, r2] += B.H2[r, r2]; end
            @inbounds for r in 1:k; B.gβ[r] += B.gβ2[r]; end
        end
        dec = _newton_dir!(B.Δ, B.H, B.gβ, β, λv, k)
        (dec > 0 && isfinite(dec)) || break
        s = one(T); ok = false
        for _ in 1:12
            @inbounds for r in 1:k; B.βt[r] = β[r] - s * B.Δ[r]; end
            _eta!(ηt, Dm, B.βt, cm, k); zi ? _eta!(ηt2, Dp, B.βt, cp, k) : copyto!(ηt2, cp)
            f1 = _pen(B.βt, λv, k)
            @inbounds for t in eachindex(a)
                iszero(w[t]) || (f1 += w[t] * _zinb_nll(a[t], ηt[t], ηt2[t], ζ[t], θ[t]))
            end
            if f1 <= f0 - 1e-4 * s * dec
                @inbounds for r in 1:k; β[r] = B.βt[r]; end
                copyto!(η, ηt); copyto!(η2, ηt2); f0 = f1; ok = true; break
            end
            s /= 2
        end
        ok || break
        step < nsteps && (f0 = lossd!() + _pen(β, λv, k))
    end
    return f0
end

# gene j: βm = [y_mu; b_mu], βp = [y_pi; b_pi] on design Xa = [X 1]; θ fixed
function _zinb_col!(βm, βp, Xa, o, a, w, ζ::T, θ::T, λv, nsteps, zi::Bool, B) where {T}
    q = size(Xa, 2); η, η2, ηt, ηt2, g, g2, h, h2 = B.η, B.η2, B.ηt, B.ηt2, B.g, B.g2, B.h, B.h2
    c2 = B.c2                                     # zeros (no offset for π)
    function lossd!()
        f = 0.0
        @inbounds for t in eachindex(a)
            if iszero(w[t]); g[t] = h[t] = g2[t] = h2[t] = zero(T); continue; end
            ℓ, gm, gp, wm, wp = _zinb_entry(a[t], η[t], η2[t], ζ, θ)
            f += w[t] * ℓ; g[t] = w[t] * gm; h[t] = w[t] * wm; g2[t] = w[t] * gp; h2[t] = w[t] * wp
        end
        f
    end
    pen(bm, bp) = _pen(bm, λv, q) + _pen(bp, λv, q)
    _eta!(η, Xa, βm, o, q); _eta!(η2, Xa, βp, c2, q)
    f0 = lossd!() + pen(βm, βp)
    for step in 1:nsteps
        _hess_grad!(B.H, B.gβ, Xa, h, g, q)
        dec = _newton_dir!(B.Δ, B.H, B.gβ, βm, λv, q)
        if zi
            _hess_grad!(B.H2, B.gβ2, Xa, h2, g2, q)
            dec += _newton_dir!(B.Δ2, B.H2, B.gβ2, βp, λv, q)
        else                                       # π: intercept only
            sg = 0.0; sh = 0.0
            @inbounds for t in eachindex(a); sg += g2[t]; sh += h2[t]; end
            fill!(B.Δ2, zero(T)); B.Δ2[q] = T(sg / max(sh, 1e-8)); dec += sg * B.Δ2[q]
        end
        (dec > 0 && isfinite(dec)) || break
        s = one(T); ok = false
        for _ in 1:12
            @inbounds for r in 1:q; B.βt[r] = βm[r] - s * B.Δ[r]; B.βt2[r] = βp[r] - s * B.Δ2[r]; end
            _eta!(ηt, Xa, B.βt, o, q); _eta!(ηt2, Xa, B.βt2, c2, q)
            f1 = pen(B.βt, B.βt2)
            @inbounds for t in eachindex(a)
                iszero(w[t]) || (f1 += w[t] * _zinb_nll(a[t], ηt[t], ηt2[t], ζ, θ))
            end
            if f1 <= f0 - 1e-4 * s * dec
                @inbounds for r in 1:q; βm[r] = B.βt[r]; βp[r] = B.βt2[r]; end
                copyto!(η, ηt); copyto!(η2, ηt2); f0 = f1; ok = true; break
            end
            s /= 2
        end
        ok || break
        step < nsteps && (f0 = lossd!() + pen(βm, βp))
    end
    return f0
end

# dispersion start: per-gene moment estimate θ = m²/(v - m) on observed counts, log θ ∈ [-2, 4]
function _init_disp(data::_Data{T}) where {T}
    n, p = size(data.A); ζ0 = zeros(T, p); a = zeros(T, n); w = zeros(T, n)
    for j in 1:p
        _getcol!(a, w, data, j); nw = max(sum(w), one(T))
        m = dot(w, a) / nw; v = dot(w, (a .- m) .^ 2) / nw
        ζ0[j] = clamp(log(m^2 / max(v - m, T(1e-2) * m^2 + eps(T)) + eps(T)), T(-2), T(4))
    end
    return ζ0
end

const _ZETA_MIN = log(1e-3)   # θ ∈ [1e-3, 1e5]
const _ZETA_MAX = log(1e5)

# safeguarded Newton on ζ_j = log θ_j given current η's (Float64 internally)
function _zinb_disp(ζ0, ηm, ηp, a, w, λd, ζbar, nsteps)
    ζ = Float64(ζ0)
    colnll(z) = (θ = exp(z); f = 0.5λd * (z - ζbar)^2;
                 @inbounds for t in eachindex(a)
                     iszero(w[t]) || (f += w[t] * _zinb_disp_entry(Float64(a[t]), Float64(ηm[t]), Float64(ηp[t]), z, θ)[1])
                 end; f)
    for _ in 1:nsteps
        θ = exp(ζ); f0 = 0.5λd * (ζ - ζbar)^2; G = λd * (ζ - ζbar); Hh = λd
        @inbounds for t in eachindex(a)
            iszero(w[t]) && continue
            ℓ, d1, d2 = _zinb_disp_entry(Float64(a[t]), Float64(ηm[t]), Float64(ηp[t]), ζ, θ)
            f0 += w[t] * ℓ; G += w[t] * d1; Hh += w[t] * d2
        end
        hh = Hh > 0 ? Hh : abs(G) + 1.0
        Δ = clamp(G / hh, -1.0, 1.0)
        abs(Δ) < 1e-6 && break
        s = 1.0; ok = false
        for _ in 1:10
            ζt = clamp(ζ - s * Δ, _ZETA_MIN, _ZETA_MAX)
            if colnll(ζt) <= f0 - 1e-4 * G * (ζ - ζt)
                ζ = ζt; ok = true; break
            end
            s /= 2
        end
        ok || break
    end
    return ζ
end

function _objective_zinb(data, Xa, Ym, Yp, ζ, o, λ, λd, bufs)
    p = size(Ym, 1); q = size(Ym, 2); k = q - 1
    foreach(B -> (B.acc[] = 0.0), bufs)
    _tforeach(p, bufs) do j, B
        _getcol!(B.a, B.w, data, j)
        @inbounds for r in 1:q; B.β[r] = Ym[j, r]; B.β2[r] = Yp[j, r]; end
        _eta!(B.η, Xa, B.β, o, q); _eta!(B.η2, Xa, B.β2, B.c2, q)
        f = 0.0
        @inbounds for t in eachindex(B.a)
            iszero(B.w[t]) || (f += B.w[t] * _zinb_nll_full(B.a[t], B.η[t], B.η2[t], ζ[j]))
        end
        B.acc[] += f
    end
    ζbar = mean(ζ)
    return sum(B -> B.acc[], bufs) + 0.5λd * sum(abs2, ζ .- ζbar) +
           0.5 * λ * (sum(abs2, view(Xa, :, 1:k)) + sum(abs2, view(Ym, :, 1:k)) + sum(abs2, view(Yp, :, 1:k)))
end

# Newton/IRLS alternating loop for ZINB (in place on Xa, Ym, Yp, ζ)
function _zinb_newton!(data, Xa, Ym, Yp, ζ, o, λ, λdisp, zi_lowrank, maxiter, tol, inner, disp_steps,
                       verbose, time_limit = Inf)
        T = eltype(Xa); n, q = size(Xa); k = q - 1; p = size(Ym, 1); t0 = time()
        nt = Threads.nthreads()
        bufX = [_Buf(T, p, q) for _ in 1:nt]; bufY = [_Buf(T, n, q) for _ in 1:nt]
        λX = fill(T(λ), q); λY = [fill(T(λ), k); zero(T)]
        θ = exp.(ζ); trace = Float64[]; conv = false
        for it in 1:maxiter
            θ .= exp.(ζ)
            _tforeach(n, bufX) do i, B                     # rows of X
                _getrow!(B.a, B.w, data, i); oi = o[i]
                @inbounds for j in 1:p; B.c[j] = Ym[j, q] + oi; B.c2[j] = Yp[j, q]; end
                @inbounds for r in 1:k; B.β[r] = Xa[i, r]; end
                _zinb_row!(B.β, Ym, Yp, B.c, B.c2, B.a, B.w, ζ, θ, k, λX, inner, zi_lowrank, B)
                @inbounds for r in 1:k; Xa[i, r] = B.β[r]; end
            end
            ζbar = mean(Float64, ζ)
            _tforeach(p, bufY) do j, B                     # genes: loadings, intercepts, dispersion
                _getcol!(B.a, B.w, data, j)
                @inbounds for r in 1:q; B.β[r] = Ym[j, r]; B.β2[r] = Yp[j, r]; end
                _zinb_col!(B.β, B.β2, Xa, o, B.a, B.w, ζ[j], θ[j], λY, inner, zi_lowrank, B)
                @inbounds for r in 1:q; Ym[j, r] = B.β[r]; Yp[j, r] = B.β2[r]; end
                ζ[j] = T(_zinb_disp(ζ[j], B.η, B.η2, B.a, B.w, Float64(λdisp), ζbar, disp_steps))
            end
            f = _objective_zinb(data, Xa, Ym, Yp, ζ, o, Float64(λ), Float64(λdisp), bufY)
            push!(trace, f)
            verbose && println("iter $it  objective $(round(f; sigdigits = 10))")
            if it > 1 && abs(trace[end-1] - f) <= tol * (abs(f) + 1)
                conv = true; break
            end
            time() - t0 > time_limit && break
        end
        return trace, conv
end

"""
    fit_zinb_pca(A, k; offset=nothing, mask=nothing, T=Float32, λ=1, λdisp=10, zi_lowrank=true,
                 maxiter=300, tol=1e-6, inner=1, disp_steps=2, init=:svd, nstarts=1, rng, verbose=false)

ZINB-WaVE-style zero-inflated negative-binomial factor model
    a_ij ~ π_ij δ₀ + (1-π_ij) NB(μ_ij, θ_j)
    log μ_ij   = o_i + b_j  + x_i' y_j          (Y  = μ loadings)
    logit π_ij =       c_j  + x_i' ypi_j        (Ypi = zero-inflation loadings)
with shared cell scores X (as in ZINB-WaVE's W) and per-gene dispersion
θ_j = exp(ζ_j) (Var = μ + μ²/θ). `zi_lowrank=false` gives π_ij = logistic(c_j)
(gene-wise zero inflation only). `λdisp` shrinks ζ_j towards their mean.
Input handling (dense / sparse / missing / mask / offset), `solver`
(:newton / :adam / :hybrid) and its keywords, and `time_limit` as in
[`fit_poisson_pca`](@ref). With :adam, ζ = log θ is a further Adam parameter.

Returns `(X, Y, Ypi, intercept, intercept_pi, dispersion, offset, d, loss,
converged, iters, family)`; [Y; Ypi] jointly orthonormal after `orthogonalize`.
"""
function fit_zinb_pca(A, k::Integer; offset = nothing, mask = nothing, T::Type{<:AbstractFloat} = Float32,
                      λ = 1.0, λdisp = 10.0, zi_lowrank::Bool = true, maxiter::Integer = 300, tol = 1e-6,
                      inner::Integer = 1, disp_steps::Integer = 2, init::Symbol = :svd, nstarts::Integer = 1,
                      rng = Random.default_rng(), verbose::Bool = false, solver::Symbol = :newton, lr = 0.02,
                      batchsize = nothing, epochs::Integer = 200, adam_epochs::Integer = epochs, polish_iters::Integer = 20,
                      adam_tol = 1e-5, time_limit = Inf)
    solver in (:newton, :adam, :hybrid) || throw(ArgumentError("solver ∈ (:newton, :adam, :hybrid)"))
    data = _prepare(A, mask, T); n, p = size(data.A); q = k + 1
    bs = batchsize === nothing ? clamp(n ÷ 16, 64, 1024) : Int(batchsize)
    0 < k < min(n, p) || throw(ArgumentError("need 0 < k < min(n, p)"))
    o = offset === nothing ? zeros(T, n) : T.(collect(offset))
    length(o) == n || throw(DimensionMismatch("offset must have length n = $n"))
    X0, Y0, b0 = _init_svd(data, o, k, false, rng)
    ζ0 = _init_disp(data)                           # π starts at 5 % (below)
    best = nothing
    for s in 1:nstarts
        if s == 1 && init === :svd
            X, Y = copy(X0), copy(Y0)
        else
            X, Y = T(0.1) .* randn(rng, T, n, k), T(0.1) .* randn(rng, T, p, k)
        end
        Xa = hcat(X, ones(T, n)); Ym = hcat(Y, copy(b0)); Yp = hcat(zeros(T, p, k), fill(T(logit(0.05)), p))
        ζ = copy(ζ0)
        t0 = time()
        if solver === :newton
            trace, conv = _zinb_newton!(data, Xa, Ym, Yp, ζ, o, λ, λdisp, zi_lowrank, maxiter, tol, inner,
                                        disp_steps, verbose, time_limit)
        else
            ne = solver === :adam ? epochs : adam_epochs
            trace, conv = _adam_zinb!(data, Xa, Ym, Yp, ζ, o, T(λ), λdisp, zi_lowrank, lr, bs, ne, adam_tol,
                                      rng, verbose, time_limit)
            if solver === :hybrid
                tr2, conv = _zinb_newton!(data, Xa, Ym, Yp, ζ, o, λ, λdisp, zi_lowrank, min(maxiter, polish_iters),
                                          tol, inner, disp_steps, verbose, time_limit - (time() - t0))
                trace = vcat(trace, tr2)
            end
        end
        if best === nothing || trace[end] < best.trace[end]
            best = (Xa = Xa, Ym = Ym, Yp = Yp, ζ = ζ, trace = trace, conv = conv)
        end
    end
    Xa, Ym, Yp, ζ = best.Xa, best.Ym, best.Yp, best.ζ
    X, Ys, bst, d = orthogonalize(Xa[:, 1:k], vcat(Ym[:, 1:k], Yp[:, 1:k]), vcat(Ym[:, q], Yp[:, q]))
    return (X = X, Y = Ys[1:p, :], Ypi = Ys[p+1:end, :], intercept = bst[1:p], intercept_pi = bst[p+1:end],
            dispersion = exp.(ζ), offset = o, d = d, loss = best.trace, converged = best.conv,
            iters = length(best.trace), family = :zinb)
end


# ----------------------------------------------------------------------------
# Mini-batch Adam (hand-written, in place; analytic gradients, BLAS gemms)
# ----------------------------------------------------------------------------
# Adam on P[:, cols] with a shared step counter t
function _adam!(P::AbstractArray{T}, G, M, V, t::Int, lr, cols) where {T}
    b1 = T(0.9); b2 = T(0.999); ϵ = T(1e-8)
    a = T(lr / (1 - 0.9^t)); c2 = T(1 - 0.999^t)
    @inbounds for r in cols
        @simd for j in axes(P, 1)
            g = G[j, r]
            m = b1 * M[j, r] + (1 - b1) * g; v = b2 * V[j, r] + (1 - b2) * g * g
            M[j, r] = m; V[j, r] = v
            P[j, r] -= a * m / (sqrt(v / c2) + ϵ)
        end
    end
end
# Adam on rows I of X (cols 1:k) with per-row step counters (lazy bias correction)
function _adam_rows!(X::AbstractMatrix{T}, G, M, V, tX, I, lr, k) where {T}
    b1 = T(0.9); b2 = T(0.999); ϵ = T(1e-8)
    @inbounds for (c, i) in enumerate(I)
        tX[i] += 1; t = tX[i]
        a = T(lr / (1 - 0.9^t)); c2 = T(1 - 0.999^t)
        for r in 1:k
            g = G[c, r]
            m = b1 * M[i, r] + (1 - b1) * g; v = b2 * V[i, r] + (1 - b2) * g * g
            M[i, r] = m; V[i, r] = v
            X[i, r] -= a * m / (sqrt(v / c2) + ϵ)
        end
    end
end

# epoch bookkeeping shared by both Adam loops: returns (lr, best, nhalf, stop)
function _adam_schedule(f, best, lr, nhalf, tol; maxhalf = 4)
    if f < best - tol * abs(best)
        return lr, f, nhalf, false
    else
        return lr / 2, min(best, f), nhalf + 1, nhalf + 1 > maxhalf
    end
end

function _adam_single!(fam, data, Xa, Ya, o, λ::T, lr, bs, epochs, tol, rng, verbose, time_limit = Inf,
                       ζ = nothing, λd = 0.0) where {T}
    n, q = size(Xa); k = q - 1; p = size(Ya, 1); bs = min(bs, n); t0 = time()
    AbT = zeros(T, p, bs); WbT = zeros(T, p, bs); GT = zeros(T, p, bs)
    Xb = zeros(T, bs, q); gXb = zeros(T, bs, q); gY = zeros(T, p, q)
    MX = zeros(T, n, q); VX = zeros(T, n, q); tX = zeros(Int, n); MY = zeros(T, p, q); VY = zeros(T, p, q); tY = 0
    nt = Threads.nthreads(); objbufs = [_Buf(T, n, q) for _ in 1:nt]; acc = [zeros(T, p) for _ in 1:nt]
    isnb = ζ !== nothing                            # NB: ζ = log θ is an extra Adam parameter
    auxv = isnb ? _nbaux(ζ) : nothing; Mz = zeros(T, p, 1); Vz = zeros(T, p, 1); gz = zeros(T, p, 1)
    Z = isnb ? reshape(ζ, p, 1) : gz
    perm = collect(1:n); trace = Float64[]; best = Inf; conv = false; nhalf = 0
    for ep in 1:epochs
        shuffle!(rng, perm)
        for s in 1:bs:n
            nb = min(bs, n - s + 1)
            (s > 1 && nb < max(2, bs ÷ 4)) && continue          # drop a tiny tail batch
            I = view(perm, s:s+nb-1)
            Gv = view(GT, :, 1:nb); Xv = view(Xb, 1:nb, :); gXv = view(gXb, 1:nb, :)
            @inbounds for r in 1:q, c in 1:nb; Xb[c, r] = Xa[I[c], r]; end
            mul!(Gv, Ya, transpose(Xv))                         # η - o  (p×nb)
            if isnb
                @inbounds for j in 1:p; auxv[j] = (ζ[j], exp(ζ[j])); end
                foreach(v -> fill!(v, zero(T)), acc)
            end
            aux = isnb ? auxv : _NOAUX
            _tforeach(nb, acc) do c, az
                i = I[c]; oi = o[i]
                _getrow!(view(AbT, :, c), view(WbT, :, c), data, i)
                @inbounds for j in 1:p
                    w = WbT[j, c]
                    if iszero(w); GT[j, c] = zero(T); continue; end
                    η = GT[j, c] + oi
                    GT[j, c] = w * _entry(fam, AbT[j, c], η, aux[j])[2]
                    isnb && (az[j] += w * _zinb_grads(AbT[j, c], η, T(-Inf), auxv[j][1], auxv[j][2])[3])
                end
            end
            mul!(gXv, transpose(Gv), Ya)                        # ∂/∂X_I
            mul!(gY, Gv, Xv)                                    # ∂/∂[Y b] from the batch
            sc = T(n / nb)
            @inbounds for r in 1:q, j in 1:p
                gY[j, r] = sc * gY[j, r] + (r <= k ? λ * Ya[j, r] : zero(T))
            end
            @inbounds for r in 1:k, c in 1:nb; gXb[c, r] += λ * Xb[c, r]; end
            tY += 1
            _adam!(Ya, gY, MY, VY, tY, lr, 1:q)
            _adam_rows!(Xa, gXb, MX, VX, tX, I, lr, k)
            if isnb
                ζbar = T(mean(ζ))
                @inbounds for j in 1:p; gz[j, 1] = sc * sum(v -> v[j], acc) + T(λd) * (ζ[j] - ζbar); end
                _adam!(Z, gz, Mz, Vz, tY, lr, 1:1)
                clamp!(ζ, T(_ZETA_MIN), T(_ZETA_MAX))
            end
        end
        f = _objective(fam, data, Xa, Ya, o, λ, objbufs, isnb ? _nbaux(ζ) : _NOAUX) + _disp_pen(ζ, λd)
        push!(trace, f)
        verbose && println("adam epoch $ep  objective $(round(f; sigdigits = 10))  lr $lr")
        lr, best, nhalf, stop = _adam_schedule(f, best, lr, nhalf, tol)
        stop && (conv = true; break)
        time() - t0 > time_limit && break
    end
    return trace, conv
end

# ZINB gradients wrt (ηm, ηp, ζ) for one entry
@inline function _zinb_grads(a::T, ηm::T, ηp::T, ζ::T, θ::T) where {T}
    u = ηm - ζ; L = log1pexp(u); r = logistic(u); pz = logistic(ηp); lq = log1pexp(ηp)
    if iszero(a)
        lp0 = logaddexp(ηp - lq, -lq - θ * L); z = exp(ηp - lq - lp0)
        return ((1 - z) * θ * r, pz - z, -(1 - z) * θ * (r - L))
    else
        dg = T(digamma(Float64(a) + Float64(θ)) - digamma(Float64(θ)))
        return (θ * r - a * (1 - r), pz, -θ * (dg - L + r - a * (1 - r) / θ))
    end
end

function _adam_zinb!(data, Xa, Ym, Yp, ζ, o, λ::T, λd, zi, lr, bs, epochs, tol, rng, verbose, time_limit = Inf) where {T}
    n, q = size(Xa); k = q - 1; p = size(Ym, 1); bs = min(bs, n); t0 = time()
    AbT = zeros(T, p, bs); WbT = zeros(T, p, bs); GmT = zeros(T, p, bs); GpT = zeros(T, p, bs)
    Xb = zeros(T, bs, q); gXb = zeros(T, bs, q); gXb2 = zeros(T, bs, q); gYm = zeros(T, p, q); gYp = zeros(T, p, q)
    MX = zeros(T, n, q); VX = zeros(T, n, q); tX = zeros(Int, n)
    MYm = zeros(T, p, q); VYm = zeros(T, p, q); MYp = zeros(T, p, q); VYp = zeros(T, p, q)
    Mz = zeros(T, p, 1); Vz = zeros(T, p, 1); gz = zeros(T, p, 1); Z = reshape(ζ, p, 1); θ = exp.(ζ); tY = 0
    nt = Threads.nthreads(); objbufs = [_Buf(T, n, q) for _ in 1:nt]; acc = [zeros(T, p) for _ in 1:nt]
    ycols = zi ? (1:q) : (q:q)
    perm = collect(1:n); trace = Float64[]; best = Inf; conv = false; nhalf = 0
    for ep in 1:epochs
        shuffle!(rng, perm)
        for s in 1:bs:n
            nb = min(bs, n - s + 1)
            (s > 1 && nb < max(2, bs ÷ 4)) && continue
            I = view(perm, s:s+nb-1)
            Gm = view(GmT, :, 1:nb); Gp = view(GpT, :, 1:nb); Xv = view(Xb, 1:nb, :)
            @inbounds for r in 1:q, c in 1:nb; Xb[c, r] = Xa[I[c], r]; end
            mul!(Gm, Ym, transpose(Xv)); mul!(Gp, Yp, transpose(Xv))
            θ .= exp.(ζ); foreach(v -> fill!(v, zero(T)), acc)
            _tforeach(nb, acc) do c, az
                i = I[c]; oi = o[i]
                _getrow!(view(AbT, :, c), view(WbT, :, c), data, i)
                @inbounds for j in 1:p
                    w = WbT[j, c]
                    if iszero(w); GmT[j, c] = zero(T); GpT[j, c] = zero(T); continue; end
                    gm, gp, gζ = _zinb_grads(AbT[j, c], GmT[j, c] + oi, GpT[j, c], ζ[j], θ[j])
                    GmT[j, c] = w * gm; GpT[j, c] = w * gp; az[j] += w * gζ
                end
            end
            gXv = view(gXb, 1:nb, :)
            mul!(gXv, transpose(Gm), Ym)
            zi && (mul!(view(gXb2, 1:nb, :), transpose(Gp), Yp); gXv .+= view(gXb2, 1:nb, :))
            mul!(gYm, Gm, Xv); mul!(gYp, Gp, Xv)
            sc = T(n / nb); ζbar = T(mean(ζ))
            @inbounds for r in 1:q, j in 1:p
                pen = r <= k ? λ : zero(T)
                gYm[j, r] = sc * gYm[j, r] + pen * Ym[j, r]
                gYp[j, r] = sc * gYp[j, r] + pen * Yp[j, r]
            end
            @inbounds for j in 1:p
                gz[j, 1] = sc * sum(a -> a[j], acc) + T(λd) * (ζ[j] - ζbar)
            end
            @inbounds for r in 1:k, c in 1:nb; gXb[c, r] += λ * Xb[c, r]; end
            tY += 1
            _adam!(Ym, gYm, MYm, VYm, tY, lr, 1:q)
            _adam!(Yp, gYp, MYp, VYp, tY, lr, ycols)
            _adam!(Z, gz, Mz, Vz, tY, lr, 1:1)
            clamp!(ζ, T(_ZETA_MIN), T(_ZETA_MAX))
            _adam_rows!(Xa, gXb, MX, VX, tX, I, lr, k)
        end
        f = _objective_zinb(data, Xa, Ym, Yp, ζ, o, Float64(λ), Float64(λd), objbufs)
        push!(trace, f)
        verbose && println("adam epoch $ep  objective $(round(f; sigdigits = 10))  lr $lr")
        lr, best, nhalf, stop = _adam_schedule(f, best, lr, nhalf, tol)
        stop && (conv = true; break)
        time() - t0 > time_limit && break
    end
    return trace, conv
end

"""
    fit_poisson_pca_adam(A, k; kw...), fit_logistic_pca_adam(A, k; kw...), fit_nb_pca_adam(A, k; kw...),
    fit_zinb_pca_adam(A, k; kw...)

Shorthands for `fit_*_pca(A, k; solver = :adam, kw...)`.
"""
fit_poisson_pca_adam(A, k::Integer; kw...) = fit_poisson_pca(A, k; solver = :adam, kw...)
fit_logistic_pca_adam(A, k::Integer; kw...) = fit_logistic_pca(A, k; solver = :adam, kw...)
fit_zinb_pca_adam(A, k::Integer; kw...) = fit_zinb_pca(A, k; solver = :adam, kw...)
fit_nb_pca_adam(A, k::Integer; kw...) = fit_nb_pca(A, k; solver = :adam, kw...)

# ----------------------------------------------------------------------------
# Helpers: linear predictor, library-size offset, held-out deviance, rank selection
# ----------------------------------------------------------------------------
"""
    linear_predictor(fit)

Dense natural parameter(s) of a fit: `η = o .+ X*Y' .+ b'` (n×p); for ZINB a
NamedTuple `(mu = log μ, pi = logit π)`.
"""
function linear_predictor(fit)
    ηm = fit.offset .+ fit.X * fit.Y' .+ fit.intercept'
    fit.family === :zinb || return ηm
    return (mu = ηm, pi = fit.X * fit.Ypi' .+ fit.intercept_pi')
end

"""
    log_libsize(A) -> Vector

Centred log library size `log(rowsum) - mean(log(rowsum))` (rows = cells), the
usual row offset for count models. Missing entries are skipped.
"""
function log_libsize(A)
    s = A isa SparseMatrixCSC ? vec(sum(A; dims = 2)) : [sum(skipmissing(r)) for r in eachrow(A)]
    l = log.(max.(Float64.(s), 1.0))
    return l .- mean(l)
end

# per-entry held-out score: Poisson/Bernoulli deviance; NB/ZINB -2 log-lik (the NB unit
# deviance depends on each model's own θ and is not comparable across ranks)
function _heldout_deviance(fit, data::_Data, test::BitMatrix)
    n, p = size(data.A); a = zeros(Float64, n); w = zeros(Float64, n)
    tot = 0.0; cnt = 0
    for j in 1:p
        _getcol!(a, w, data, j)
        ηm = fit.offset .+ fit.X * fit.Y[j, :] .+ fit.intercept[j]
        ηp = fit.family === :zinb ? fit.X * fit.Ypi[j, :] .+ fit.intercept_pi[j] : ηm
        for i in 1:n
            test[i, j] || continue
            y = a[i]; e = Float64(ηm[i])
            tot += fit.family === :poisson   ? 2 * ((y > 0 ? y * (log(y) - e) : 0.0) - (y - exp(e))) :
                   fit.family === :bernoulli ? 2 * (log1pexp(e) - y * e) :
                   fit.family === :nb        ? 2 * _nll_full(NBFamily(), y, e, (log(fit.dispersion[j]), 0.0)) :
                   2 * _zinb_nll_full(y, e, ηp[i], log(fit.dispersion[j]))
            cnt += 1
        end
    end
    return tot / max(cnt, 1)
end

"""
    select_rank(A, ks; family=:poisson, holdout=0.1, mask=nothing, rng, kwargs...)

Rank selection by held-out deviance: a random fraction `holdout` of the
observed entries is masked, each `k in ks` is fitted on the rest (kwargs are
passed to the fitter, e.g. `offset`, `λ`, `T`) and the mean per-entry deviance
on the held-out entries is computed (Poisson/Bernoulli: deviance; NB/ZINB:
-2 log-lik, since the NB deviance depends on the fitted θ). `family ∈ (:poisson, :bernoulli, :nb, :zinb)`. Returns `(ks, deviance, best_k, fits)`.
"""
function select_rank(A, ks; family::Symbol = :poisson, holdout = 0.1, mask = nothing,
                     rng = Random.default_rng(), kwargs...)
    fitter = family === :poisson ? fit_poisson_pca : family === :bernoulli ? fit_logistic_pca :
             family === :nb ? fit_nb_pca : family === :zinb ? fit_zinb_pca :
             throw(ArgumentError("family ∈ (:poisson, :bernoulli, :nb, :zinb)"))
    data = _prepare(A, mask, Float64); n, p = size(data.A)
    obs = data.mask === nothing ? trues(n, p) : data.mask
    test = BitMatrix(rand(rng, n, p) .< holdout) .& obs
    train = obs .& .!test
    fits = [fitter(A, k; mask = train, rng, kwargs...) for k in ks]
    dev = [_heldout_deviance(f, data, test) for f in fits]
    return (ks = collect(ks), deviance = dev, best_k = collect(ks)[argmin(dev)], fits = fits)
end

"""
    max_subspace_angle(U, V) -> degrees

Largest principal angle between the column spaces of `U` and `V`
(0° = identical subspaces). Handy to compare true vs fitted loadings.
"""
function max_subspace_angle(U, V)
    Qu = Matrix(qr(Float64.(U)).Q); Qv = Matrix(qr(Float64.(V)).Q)
    return acosd(clamp(minimum(svdvals(Qu' * Qv)), 0.0, 1.0))
end

# ----------------------------------------------------------------------------
# Simulation (no Distributions.jl: Marsaglia–Tsang gamma, inversion/PTRS Poisson)
# ----------------------------------------------------------------------------
function _rand_gamma(rng, α::Float64)              # Gamma(α, 1)
    α < 1 && return _rand_gamma(rng, α + 1) * rand(rng)^(1 / α)
    d = α - 1 / 3; c = 1 / sqrt(9d)
    while true
        x = randn(rng); v = (1 + c * x)^3
        v <= 0 && continue
        log(rand(rng)) < 0.5x^2 + d - d * v + d * log(v) && return d * v
    end
end
function _rand_poisson(rng, λ::Float64)
    λ <= 0 && return 0
    if λ < 30                                      # inversion
        k = 0; pk = exp(-λ); s = pk; u = rand(rng)
        while u > s && k < 10_000; k += 1; pk *= λ / k; s += pk; end
        return k
    end
    slam = sqrt(λ); ll = log(λ)                    # PTRS (Hörmann 1993)
    b = 0.931 + 2.53slam; a = -0.059 + 0.02483b; ia = 1.1239 + 1.1328 / (b - 3.4); vr = 0.9277 - 3.6224 / (b - 2)
    while true
        U = rand(rng) - 0.5; V = rand(rng); us = 0.5 - abs(U)
        k = floor(Int, (2a / us + b) * U + λ + 0.43)
        (us >= 0.07 && V <= vr) && return k
        (k < 0 || (us < 0.013 && V > us)) && continue
        log(V) + log(ia) - log(a / us^2 + b) <= -λ + k * ll - loggamma(k + 1.0) && return k
    end
end

"""
    simulate_poisson(n, p, k; T=Float32, signal=1.0, base=0.5, libsd=0.3, rng)

Simulate Poisson low-rank counts. Returns `(A, η, X, Y, b, offset)` with
`η = offset .+ X*Y' .+ b'` and `A ~ Poisson(exp(η))` (dense Matrix{T}).
"""
function simulate_poisson(n, p, k; T = Float32, signal = 1.0, base = 0.5, libsd = 0.3, rng = Random.default_rng())
    X = randn(rng, n, k); Y = randn(rng, p, k) .* (signal / sqrt(k))
    b = base .+ 0.5 .* randn(rng, p); o = libsd .* randn(rng, n)
    η = o .+ X * Y' .+ b'
    A = T.(map(e -> _rand_poisson(rng, exp(e)), η))
    return (A = A, η = η, X = X, Y = Y, b = b, offset = o)
end

"""
    simulate_binary(n, p, k; T=Float32, signal=2.0, rng)

Simulate logistic low-rank binary data: `A ~ Bernoulli(logistic(X*Y' .+ b'))`.
Returns `(A, η, X, Y, b)`.
"""
function simulate_binary(n, p, k; T = Float32, signal = 2.0, rng = Random.default_rng())
    X = randn(rng, n, k); Y = randn(rng, p, k) .* (signal / sqrt(k)); b = 0.5 .* randn(rng, p)
    η = X * Y' .+ b'
    A = T.(rand(rng, n, p) .< logistic.(η))
    return (A = A, η = η, X = X, Y = Y, b = b)
end

"""
    simulate_nb(n, p, k; T=Float32, signal=1.0, base=0.5, theta_mean=3.0, libsd=0.3, rng)

Simulate NB2 low-rank counts: `log μ = offset .+ X*Y' .+ b'`, θ_j log-normal
around `theta_mean`, `A ~ NB(μ, θ)` (gamma–Poisson). Returns `(A, η, X, Y, b, theta, offset)`.
"""
function simulate_nb(n, p, k; T = Float32, signal = 1.0, base = 0.5, theta_mean = 3.0, libsd = 0.3,
                     rng = Random.default_rng())
    X = randn(rng, n, k); Y = randn(rng, p, k) .* (signal / sqrt(k))
    b = base .+ 0.5 .* randn(rng, p); o = libsd .* randn(rng, n); θ = theta_mean .* exp.(0.5 .* randn(rng, p))
    η = o .+ X * Y' .+ b'
    A = zeros(T, n, p)
    for j in 1:p, i in 1:n
        A[i, j] = _rand_poisson(rng, exp(η[i, j]) * _rand_gamma(rng, θ[j]) / θ[j])
    end
    return (A = A, η = η, X = X, Y = Y, b = b, theta = θ, offset = o)
end

"""
    simulate_zinb(n, p, k; T=Float32, signal=1.0, base=1.0, pi_base=-1.5, pi_signal=0.5,
                  theta_mean=3.0, libsd=0.3, rng)

Simulate ZINB low-rank counts with shared scores X:
`log μ = o .+ X*Y' .+ b'`, `logit π = X*Ypi' .+ c'`, θ_j log-normal around `theta_mean`.
Returns `(A, ηmu, ηpi, X, Y, Ypi, b, c, theta, offset)`.
"""
function simulate_zinb(n, p, k; T = Float32, signal = 1.0, base = 1.0, pi_base = -1.5, pi_signal = 0.5,
                       theta_mean = 3.0, libsd = 0.3, rng = Random.default_rng())
    X = randn(rng, n, k); Y = randn(rng, p, k) .* (signal / sqrt(k)); Ypi = randn(rng, p, k) .* (pi_signal / sqrt(k))
    b = base .+ 0.5 .* randn(rng, p); c = pi_base .+ 0.5 .* randn(rng, p); o = libsd .* randn(rng, n)
    θ = theta_mean .* exp.(0.5 .* randn(rng, p))
    ηm = o .+ X * Y' .+ b'; ηp = X * Ypi' .+ c'
    A = zeros(T, n, p)
    for j in 1:p, i in 1:n
        rand(rng) < logistic(ηp[i, j]) && continue
        A[i, j] = _rand_poisson(rng, exp(ηm[i, j]) * _rand_gamma(rng, θ[j]) / θ[j])
    end
    return (A = A, ηmu = ηm, ηpi = ηp, X = X, Y = Y, Ypi = Ypi, b = b, c = c, theta = θ, offset = o)
end

# =============================================================================
# Example (runs only when executed as a script: julia -t auto --project=. src/lowrank_glm.jl)
# =============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    rng = Xoshiro(2026)
    n, p, k = 2000, 500, 5
    println("threads = $(Threads.nthreads()),  n×p = $n×$p,  k = $k")
    report(name, t, f, ηtrue, Ytrue) = println(rpad(name, 28),
        " time $(round(t; digits = 2)) s | iters $(f.iters) conv=$(f.converged) | ",
        "cor(η) = $(round(cor(vec(ηtrue), vec(Float64.(f.family === :zinb ? linear_predictor(f).mu : linear_predictor(f)))); digits = 4)) | ",
        "max angle(Y) = $(round(max_subspace_angle(Ytrue, f.Y); digits = 2))°")

    # warm-up (compilation) on tiny data
    let s = simulate_poisson(60, 40, 2; rng), z = simulate_zinb(60, 40, 2; rng), bsim = simulate_binary(60, 40, 2; rng)
        Am = Matrix{Union{Missing,Float32}}(s.A); Am[1, 1] = missing
        fit_poisson_pca(s.A, 2; offset = s.offset); fit_poisson_pca(sparse(s.A), 2; offset = s.offset)
        fit_poisson_pca(s.A, 2; offset = s.offset, T = Float64); fit_poisson_pca(Am, 2; offset = s.offset)
        fit_logistic_pca(bsim.A, 2); fit_zinb_pca(z.A, 2; offset = z.offset)
        nbs = simulate_nb(60, 40, 2; rng)
        for sv in (:newton, :adam); fit_nb_pca(nbs.A, 2; offset = nbs.offset, solver = sv); end
        for sv in (:adam, :hybrid)
            fit_poisson_pca(s.A, 2; offset = s.offset, solver = sv); fit_logistic_pca(bsim.A, 2; solver = sv)
            fit_zinb_pca(z.A, 2; offset = z.offset, solver = sv)
        end
        select_rank(s.A, 1:2; offset = s.offset, rng)
    end

    # Poisson PCA (dense, Float32), sparse input, Float64, 10 % missing
    sp = simulate_poisson(n, p, k; rng)
    t = @elapsed f = fit_poisson_pca(sp.A, k; offset = sp.offset)
    report("Poisson dense Float32", t, f, sp.η, sp.Y)
    t = @elapsed f = fit_poisson_pca(sparse(sp.A), k; offset = sp.offset)
    report("Poisson sparse Float32", t, f, sp.η, sp.Y)
    t = @elapsed f = fit_poisson_pca(sp.A, k; offset = sp.offset, T = Float64)
    report("Poisson dense Float64", t, f, sp.η, sp.Y)
    Am = Matrix{Union{Missing,Float32}}(sp.A); Am[rand(rng, n, p) .< 0.1] .= missing
    t = @elapsed f = fit_poisson_pca(Am, k; offset = sp.offset)
    report("Poisson 10% missing", t, f, sp.η, sp.Y)
    t = @elapsed f = fit_poisson_pca(sp.A, k; offset = sp.offset, solver = :adam)
    report("Poisson Adam", t, f, sp.η, sp.Y)
    t = @elapsed f = fit_poisson_pca(sparse(sp.A), k; offset = sp.offset, solver = :hybrid)
    report("Poisson sparse hybrid", t, f, sp.η, sp.Y)

    # Logistic PCA
    sb = simulate_binary(n, p, k; rng)
    t = @elapsed f = fit_logistic_pca(sb.A, k)
    report("Logistic Float32", t, f, sb.η, sb.Y)
    t = @elapsed f = fit_logistic_pca(sb.A, k; solver = :adam)
    report("Logistic Adam", t, f, sb.η, sb.Y)

    # NB PCA
    sn = simulate_nb(n, p, k; rng)
    for sv in (:newton, :adam)
        local t, f
        t = @elapsed f = fit_nb_pca(sn.A, k; offset = sn.offset, solver = sv)
        report("NB $sv", t, f, sn.η, sn.Y)
        println(" "^28, " cor(log θ) = $(round(cor(log.(sn.theta), log.(f.dispersion)); digits = 3))")
    end

    # ZINB PCA
    sz = simulate_zinb(n, p, k; rng)
    println("ZINB data: zero fraction $(round(mean(sz.A .== 0); digits = 3))")
    t = @elapsed f = fit_zinb_pca(sz.A, k; offset = sz.offset)
    report("ZINB Float32", t, f, sz.ηmu, sz.Y)
    println(" "^28, " cor(logit π) = $(round(cor(vec(sz.ηpi), vec(linear_predictor(f).pi)); digits = 3)) | ",
            "cor(log θ) = $(round(cor(log.(sz.theta), log.(f.dispersion)); digits = 3))")
    t = @elapsed f = fit_zinb_pca(sz.A, k; offset = sz.offset, solver = :adam)
    report("ZINB Adam", t, f, sz.ηmu, sz.Y)

    # rank selection by held-out deviance (smaller problem)
    ss = simulate_poisson(600, 200, 3; rng)
    t = @elapsed rs = select_rank(ss.A, 1:6; family = :poisson, offset = ss.offset, rng)
    println("select_rank (true k = 3): deviance = $(round.(rs.deviance; digits = 4)) -> best k = $(rs.best_k)  ($(round(t; digits = 1)) s)")
end
