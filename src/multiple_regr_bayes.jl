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

# Print correlation matrix
ρ_matrix = covariance_to_correlation(Σ_matrix)
println("\nCorrelation Matrix (ρ):")
println(ρ_matrix)

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
    Σ ~ LKJCholesky(q, 2.0) * Diagonal(filldist(Exponential(1), q))

    # Likelihood
    for i in 1:n
        Y[i, :] ~ MvNormal(X[i, :] * B, Σ)  # Multivariate normal likelihood
    end
end

# Step 4: Sample from the model
# Use NUTS sampler (default) for Bayesian inference
chain = sample(multivariate_regression(X, Y), NUTS(1000, 0.65), 2000)

# Step 5: Extract results
# Coefficient matrix B (mean of posterior)
B_mean = mean(chain)[:B]  # Extract mean of B parameters
B_matrix = reshape(B_mean, size(X, 2), size(Y, 2))  # Reshape to (p × q)

# Covariance matrix Σ (mean of posterior)
Σ_mean = mean(chain)[:Σ]  # Extract mean of Σ
Σ_matrix = Matrix(Σ_mean)  # Convert to matrix if needed

# Step 6: Print results
println("Coefficient Matrix (B):")
println(B_matrix)
println("\nError Covariance Matrix (Σ):")
println(Σ_matrix)

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
# Example for B[1,1] (first covariate, first outcome)
plot(chain[:B][1,1], title="Posterior of B[1,1]", xlabel="Value", ylabel="Density")

using Plots, Statistics, Random
x = collect(0:0.01:5)
y = 2*sin.(2 .* x) + 0.2 .* randn(length(x))
scatter(x,y)
cor(x,y)
xi_corr(x,y)
perm_xi_correlation_test(x,y)