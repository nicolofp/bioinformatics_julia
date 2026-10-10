# xi correlation coefficient and tests (see xi_correlation.qmd)

using Distributions, StatsBase, Statistics, Random

function xi_corr(x::AbstractVector, y::AbstractVector; rng::AbstractRNG = Random.default_rng())

    n = length(x)

    length(y) == n || throw(DimensionMismatch("x and y don't have the same length"))
    n >= 2 || throw(ArgumentError("need at least 2 observations"))

    # sort by x, breaking ties in x uniformly at random:
    # random pre-shuffle followed by a stable sort
    perm = randperm(rng, n)
    ord = perm[sortperm(view(x, perm))]
    ys = y[ord]

    r = n + 1 .- competerank(ys, rev = true)   # r_i = #{j : y_j <= y_i}
    l = n + 1 .- competerank(ys)               # l_i = #{j : y_j >= y_i}

    den = 2 * sum(l .* (n .- l))
    den == 0 && throw(ArgumentError("y is constant: xi is undefined"))
    num = n * sum(abs, diff(r))

    return 1 - num / den
end

# asymptotic variance of sqrt(n) * xi_n under independence
function xi_null_variance(y::AbstractVector)
    n = length(y)
    allunique(y) && return 2 / 5                        # continuous Y
    u = sort(n + 1 .- competerank(y, rev = true)) ./ n  # sorted r_i / n
    i = 1:n
    w = 2n .- 2 .* i .+ 1
    a = mean(w .* u .^ 2) / n
    c = mean(w .* u) / n
    m = (cumsum(u) .+ (n .- i) .* u) ./ n
    b = mean(m .^ 2)
    g = (n + 1 .- competerank(y)) ./ n                  # l_i / n
    d = mean(g .* (1 .- g))
    return (a - 2b + c^2) / d^2
end

# asymptotic test, one-sided (large xi)
function xi_test(x, y; rng::AbstractRNG = Random.default_rng())
    n = length(x)
    xi = xi_corr(x, y; rng = rng)
    tau2 = xi_null_variance(y)
    z = sqrt(n) * xi / sqrt(tau2)
    return (xi = xi, sd = sqrt(tau2 / n), z = z, pvalue = ccdf(Normal(), z))
end

function perm_xi_correlation_test(x, y; n_permutations = 10000,
                                  rng::AbstractRNG = Random.default_rng())
    # Check that vectors have same length
    length(x) != length(y) && throw(ArgumentError("x and y must have same length"))

    # Calculate observed correlation
    xi_observed = xi_corr(x, y; rng = rng)

    # Generate null distribution by permuting y
    null_correlations = Vector{Float64}(undef, n_permutations)
    for i in 1:n_permutations
        null_correlations[i] = xi_corr(x, shuffle(rng, y); rng = rng)
    end

    # One-sided p-value: large values of xi are evidence of dependence
    extreme_count = count(>=(xi_observed), null_correlations)
    p_value = (extreme_count + 1) / (n_permutations + 1)

    return xi_observed, p_value, null_correlations
end
