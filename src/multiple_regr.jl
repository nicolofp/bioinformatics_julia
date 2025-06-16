using DataFrames, GLM, MultivariateStats, Statistics, LinearAlgebra, Plots

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
Σ = (residuals' * residuals) / (N - size(X, 2))  # Covariance of errors

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



