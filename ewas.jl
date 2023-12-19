using DataFrames, Statistics, LinearAlgebra 
using Distributions, StatsBase, Random, DuckDB

# Dataset Test
#= rng = MersenneTwister(2090);
x = rand(rng,Normal(0.0,0.1),2000,5)
a = -1.2
b = vec([1.1 -0.12 0.34 2.13 -1.8])
b2 = vec([2.1 -1.12 0.94 -1.13 -0.38])
mmix = a .+ x * b
mmix2 = a .+ x * b2
K = 2000
y = rand(rng,MvNormal(mmix,0.5 * I),K)
dt = DataFrame(hcat(y,x), 
               vcat("y" .* string.(1:K),
                    "x" .* string.(1:5)))
out = "y" .* string.(1:K);
cov = "x" .* string.(1:5)=#

con = DBInterface.connect(DuckDB.DB, ":memory:")
dt = DataFrame(DBInterface.execute(con,
           """
           SELECT *
           FROM 'C:/Users/nicol/Documents/dt_limma_test.csv'
           """));

out = names(dt)[1:20]
cov = names(dt)[21:22]

@time ewas_lm(dt,out,cov,true)
@time lm_series(dt,out,cov)

function linreg3(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    (N = length(x)) == length(y) || throw(DimensionMismatch())
    ldiv!(cholesky!(Symmetric([T(N) sum(x); zero(T) sum(abs2, x)], :U)), [sum(y), dot(x, y)])
end

function linreg2(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    X = [ones(length(x)) x]
    ldiv!(cholesky!(Symmetric(X'X, :U)), X'y)
end

# Plot-twist --> I can handle multiple regression in once
function lm_series(dt, out, cov)
    X_tmp = hcat(ones(size(dt,1)), Matrix{Float64}(dt[:,cov]))
    Y_tmp = Matrix{Float64}(dt[:,out])
    β = X_tmp\Y_tmp 
    σ = sqrt.(vec(sum((Y_tmp - X_tmp*β).^2,dims = 1)./(size(X_tmp,1)-size(X_tmp,2))))
    Σ = inv(X_tmp'*X_tmp)
    std_coeff_unscaled = sqrt.(diag(Σ))
    tmp = (beta = β, sigma = σ, S = Σ, std_coeff_unscaled = std_coeff_unscaled)
    return tmp
end 



         


# EWAS function 
function ewas_lm(dt, out, cov, multithreads = true)
    X_tmp = hcat(ones(size(dt,1)), Matrix{Float64}(dt[:,cov]))
    m = length(out)
    results = DataFrame(outcome = out,
                        beta = zeros(m),
                        sd = zeros(m),
                        tval = zeros(m),
                        pval = zeros(m),
                        ci025 = zeros(m),
                        ci975 = zeros(m))
    if multithreads == false
        for i in 1:length(out)
            Y_tmp = convert.(Float64, dt[:,out[i]])
            β = X_tmp\Y_tmp
            σ² = sum((Y_tmp - X_tmp*β).^2)/(size(X_tmp,1)-size(X_tmp,2))
            Σ = σ²*inv(X_tmp'*X_tmp)
            std_coeff = sqrt.(diag(Σ))

            results.beta[i] = β[2]
            results.sd[i] = std_coeff[2]
            results.tval[i] = β[2]/std_coeff[2]
            results.pval[i] = cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs(β[2]/std_coeff[2]))
            results.ci025[i] = β[2] - quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
            results.ci975[i] = β[2] + quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]            
        end  
     else
        Threads.@threads for i in 1:length(out)
            Y_tmp = convert.(Float64, dt[:,out[i]])
            β = X_tmp\Y_tmp
            σ² = sum((Y_tmp - X_tmp*β).^2)/(size(X_tmp,1)-size(X_tmp,2))
            Σ = σ²*inv(X_tmp'*X_tmp)
            std_coeff = sqrt.(diag(Σ))

            results.beta[i] = β[2]
            results.sd[i] = std_coeff[2]
            results.tval[i] = β[2]/std_coeff[2]
            results.pval[i] = cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs(β[2]/std_coeff[2]))
            results.ci025[i] = β[2] - quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
            results.ci975[i] = β[2] + quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]            
        end
    end
    return(results)
end

# eBayes function
# Todo