# High-throughput linear model for EWAS (see hp_lm.qmd)

using DataFrames, Distributions, LinearAlgebra, Statistics

# Benjamini–Hochberg adjusted p-values (same as R's p.adjust(p, "BH"))
function bh_adjust(p::AbstractVector{<:Real})
    m = length(p)
    adj = similar(p, Float64)
    m == 0 && return adj
    order = sortperm(p, rev = true)      # largest p-value first
    running = 1.0
    for (r, idx) in enumerate(order)
        rank = m - r + 1                 # rank in increasing order
        running = min(running, p[idx] * m / rank)
        adj[idx] = running
    end
    return adj
end

function lm_ewas(dt, out, cov; chunk::Integer = 10_000)
    chunk > 0 || throw(ArgumentError("`chunk` must be a positive integer"))

    # --- complete data required --------------------------------------
    for c in vcat(cov, out)
        nmiss = count(ismissing, dt[!, c])
        if nmiss > 0
            throw(ArgumentError("column `$c` has $nmiss missing value(s). " *
                "lm_ewas requires complete data for all covariates and outcomes " *
                "(the same design matrix is shared by every outcome). " *
                "Impute the missing values or filter the rows (e.g. dropmissing) before calling lm_ewas."))
        end
    end

    # --- design matrix and its QR factorization (computed once) ------
    n = size(dt, 1)
    X = hcat(ones(n), Matrix{Float64}(dt[:, cov]))
    p = size(X, 2)
    df = n - p
    df > 0 || throw(ArgumentError("need more samples (n = $n) than parameters (p = $p)"))
    F = qr(X)
    R = UpperTriangular(F.R)
    rdiag = abs.(diag(R))
    if minimum(rdiag) <= p * eps() * maximum(rdiag)
        throw(ArgumentError("the design matrix is rank deficient (collinear covariates?)"))
    end
    Rinv = inv(R)
    d = vec(sum(abs2, Rinv, dims = 2))   # diag((X'X)^-1) = diag(R^-1 R^-T)

    tdist = TDist(df)
    tq = quantile(tdist, 0.975)

    # --- outputs (long format, outcome-major, intercept first) -------
    N = length(out)
    K = N * p
    βvec = Vector{Float64}(undef, K)
    std_coeff = Vector{Float64}(undef, K)

    # --- process the outcomes in blocks, reusing F --------------------
    for start in 1:chunk:N
        cols = start:min(start + chunk - 1, N)
        Y = Matrix{Float64}(dt[:, out[cols]])
        β = F \ Y                                         # p × length(cols)
        s2 = vec(sum(abs2, Y .- X * β, dims = 1)) ./ df   # residual variances σ²
        rows = ((first(cols) - 1) * p + 1):(last(cols) * p)
        βvec[rows] .= vec(β)
        std_coeff[rows] .= sqrt.(kron(s2, d))             # Kronecker product
    end

    tval = βvec ./ std_coeff
    pval = 2 .* ccdf.(tdist, abs.(tval))                  # two-sided
    ci025 = βvec .- tq .* std_coeff
    ci975 = βvec .+ tq .* std_coeff

    # BH adjustment per covariate across outcomes; intercept not adjusted
    padj = fill(NaN, K)
    for k in 2:p
        idx = k:p:K
        padj[idx] .= bh_adjust(pval[idx])
    end

    outcome = repeat(out, inner = p)
    covariates = repeat(vcat("intercept", cov), outer = N)

    return (outcome = outcome, covariates = covariates,
            beta = βvec, sd = std_coeff, tval = tval,
            pval = pval, padj = padj, ci025 = ci025, ci975 = ci975)
end
