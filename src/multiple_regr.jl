# multiple_regr.jl - Multivariate OLS sandbox on simulated data (N=1000, p=5 covariates, q=4 outcomes):
# backslash vs MultivariateStats.llsq, residual covariance, per-outcome R^2; then a nearest-neighbour
# lookup with correlation distance on a small mixed simulated matrix. No external data.
# Packages: DataFrames, GLM, MultivariateStats, Statistics, LinearAlgebra, Plots, Random, Distributions,
#           StatsBase, Distances

using DataFrames, GLM, MultivariateStats, Statistics, LinearAlgebra, Plots
using Random, Distributions, StatsBase  # fix: Uniform/Bernoulli/sample were used without being loaded
Random.seed!(123)  # fix: reproducible simulation

# https://discourse.julialang.org/t/multivariate-ols/9150/6
q = 4
p = 5
N = 1000
X = rand(N,p)
A0, b0 = rand(p,q), rand(1, q)
Y = (X * A0 .+ b0) + 0.1 * randn(N, q)

# solve using llsq
#sol = llsq(X, Y; dims = 2)
vcat(A0,b0)
@time B1 = hcat(X,ones(1000))\Y
@time B2 = llsq(X,Y)

# Residuals
Y_pred = hcat(X,ones(1000)) * B2
residuals = Y - Y_pred

# Estimate error covariance matrix Σ
Σ = (residuals' * residuals) / (N - size(X, 2) - 1)  # Covariance of errors; fix: df = N - (p + 1) incl. intercept

# Step 6: Optional - Hypothesis testing (e.g., Wilks' Lambda)
# For simplicity, compute R² for each outcome
R2 = 1 .- var(residuals, dims=1) ./ var(Y, dims=1)

# Step 7: Diagnostics (basic)
# Check for multivariate normality of residuals (optional, using simple stats)
println("\nResidual means (should be ~0):")
println(mean(residuals, dims=1))

# If you want to visualize residuals
scatter(residuals[:,1], residuals[:,2], xlabel="Residuals Y1", 
        ylabel="Residuals Y2", title="Residual Scatter")

using Distances

X = hcat(rand(Uniform(20,50),500),
         rand(Bernoulli(0.5),500),
         sample([1,2,3],500))
X_scaled = (X .- mean(X, dims=1)) #./ std(X, dims=1)

R = pairwise(CorrDist(), X, dims=1)

i = 34
# Fix: argmin over the vector without row i returns a shifted index for rows after i
j = argmin(R[i,vcat(1:(i-1),(i+1):500)]); j = j >= i ? j + 1 : j
X[[i,j],:]
