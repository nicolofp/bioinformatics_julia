using DataFrames, GLM, MultivariateStats, Statistics, LinearAlgebra, Plots

# https://discourse.julialang.org/t/multivariate-ols/9150/6
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

# Solve using X \\ y
b = X \ Y



# Step 3: Fit multivariate regression using MultivariateStats.jl
# Fit the model: Y = XB + E
llsq(X, Y, dims=2)
LinearModel = @load LinearModel pkg=MultivariateStats;
model = fit(LinearModel, X, Y)

# Step 4: Extract results
# Coefficients (B matrix: (p+1) × q)
B = coef(model)  # Each column corresponds to one outcome

# Residuals
Y_pred = X * B
residuals = Y - Y_pred

# Estimate error covariance matrix Σ
n, q = size(Y)
Σ = (residuals' * residuals) / (n - size(X, 2))  # Covariance of errors

# Step 5: Print results
println("Coefficient Matrix (B):")
println(B)
println("\nError Covariance Matrix (Σ):")
println(Σ)

# Step 6: Optional - Hypothesis testing (e.g., Wilks' Lambda)
# For simplicity, compute R² for each outcome
R2 = 1 .- var(residuals, dims=1) ./ var(Y, dims=1)
println("\nR² for each outcome:")
println(R2)

# Step 7: Diagnostics (basic)
# Check for multivariate normality of residuals (optional, using simple stats)
println("\nResidual means (should be ~0):")
println(mean(residuals, dims=1))

# If you want to visualize residuals
scatter(residuals[:,1], residuals[:,2], xlabel="Residuals Y1", 
        ylabel="Residuals Y2", title="Residual Scatter")




X = rand(3, 1000)
A0, b0 = rand(3, 5), rand(1, 5)
Y = (X' * A0 .+ b0) + 0.1 * randn(1000, 5)

# solve using llsq
sol = llsq(X, Y; dims = 2)
vcat(A0,b0)