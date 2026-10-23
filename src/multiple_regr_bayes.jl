# multiple_regr_bayes.jl - Bayesian multivariate linear regression (Turing, NUTS) on simulated data
# (n=100, 2 outcomes, 2 covariates + intercept): posterior means of B and the error covariance Σ,
# R^2, residual plots; ends with a small xi-correlation demo using src/xi_correlation.jl.
# No external data. Packages: Turing, DataFrames, Statistics, LinearAlgebra, Plots, Random
#                             (+ Distributions, StatsBase via xi_correlation.jl)

# Compute correlation matrix from Σ
function covariance_to_correlation(Σ)
    q = size(Σ, 1)
    ρ = zeros(q, q)
    for i in 1:q
        for j in 1:q
            ρ[i,j] = Σ[i,j] / sqrt(Σ[i,i] * Σ[j,j])
        end
    end
    return ρ
end

# (correlation printout moved after Σ_matrix is estimated - fix: it was used before being defined)

using Turing, DataFrames, Statistics, LinearAlgebra, Plots, Random

# Set random seed for reproducibility
Random.seed!(123)

# Step 1: Create or load your dataset
# Example: Simulated data with 100 observations, 2 outcomes, 2 covariates
n = 100
data = DataFrame(
    Y1 = randn(n) .+ 2,  # Outcome 1 (e.g., math score)
    Y2 = randn(n) .+ 1,  # Outcome 2 (e.g., reading score)
    X1 = randn(n),       # Covariate 1 (e.g., study hours)
    X2 = randn(n)        # Covariate 2 (e.g., parental education)
)

# Step 2: Prepare the data
# Extract outcome matrix Y (n × q)
Y = Matrix(data[!, [:Y1, :Y2]])  # q = 2 outcomes

# Create design matrix X (n × (p+1)), including intercept
X = hcat(ones(n), Matrix(data[!, [:X1, :X2]]))  # p = 2 covariates + intercept

# Step 3: Define the Turing model
@model function multivariate_regression(X, Y)
    n, p = size(X)  # Number of observations, predictors (including intercept)
    q = size(Y, 2)  # Number of outcomes

    # Priors
    # Coefficients: B is a (p × q) matrix
    B ~ filldist(Normal(0, 10), p, q)  # Weakly informative prior for coefficients
    # Covariance matrix: LKJ prior for correlation matrix, vague prior for variances
    # Fix: a distribution cannot be multiplied by another; sample SDs and an LKJ correlation separately
    σ ~ filldist(Exponential(1), q)
    Ω ~ LKJ(q, 2.0)
    Σ = Symmetric(Diagonal(σ) * Ω * Diagonal(σ))

    # Likelihood
    for i in 1:n
        Y[i, :] ~ MvNormal(B' * X[i, :], Σ)  # Multivariate normal likelihood; fix: mean is B'x_i (vector * matrix errored)
    end
end

# Step 4: Sample from the model
# Use NUTS sampler (default) for Bayesian inference
chain = sample(multivariate_regression(X, Y), NUTS(1000, 0.65), 2000)

# Step 5: Extract results
# Coefficient matrix B (mean of posterior)
# Fix: mean(chain)[:B] does not work for array parameters; average the B[i,j] columns (column-major)
B_mean = vec(mean(Array(group(chain, :B)), dims = 1))
B_matrix = reshape(B_mean, size(X, 2), size(Y, 2))  # Reshape to (p × q)

# Covariance matrix Σ (mean of posterior)
# Fix: Σ is not a sampled parameter any more; rebuild it from posterior means of σ and Ω (plug-in estimate)
σ_mean = vec(mean(Array(group(chain, :σ)), dims = 1))
Ω_mean = reshape(vec(mean(Array(group(chain, :Ω)), dims = 1)), size(Y, 2), size(Y, 2))
Σ_matrix = Matrix(Diagonal(σ_mean) * Ω_mean * Diagonal(σ_mean))

# Step 6: Print results
println("Coefficient Matrix (B):")
println(B_matrix)
println("\nError Covariance Matrix (Σ):")
println(Σ_matrix)

# Print correlation matrix
ρ_matrix = covariance_to_correlation(Σ_matrix)
println("\nCorrelation Matrix (ρ):")
println(ρ_matrix)

# Step 7: Diagnostics
# Predicted values
Y_pred = X * B_matrix
residuals = Y - Y_pred

# R² for each outcome
R2 = 1 .- var(residuals, dims=1) ./ var(Y, dims=1)
println("\nR² for each outcome:")
println(R2)

# Residual means
println("\nResidual means (should be ~0):")
println(mean(residuals, dims=1))

# Visualize residuals
scatter(residuals[:,1], residuals[:,2], xlabel="Residuals Y1", ylabel="Residuals Y2", title="Residual Scatter")

# Optional: Plot posterior distributions of coefficients
# Example for B[1,1] (intercept, first outcome); fix: chain[:B][1,1] is not valid MCMCChains indexing
plot(chain[Symbol("B[1,1]")], title="Posterior of B[1,1]", xlabel="Value", ylabel="Density")

using Plots, Statistics, Random
include("xi_correlation.jl")  # fix: xi_corr / perm_xi_correlation_test are defined there
x = collect(0:0.01:5)
y = 2*sin.(2 .* x) + 0.2 .* randn(length(x))
scatter(x,y)
cor(x,y)
xi_corr(x,y)
perm_xi_correlation_test(x,y)
