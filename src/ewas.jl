# ewas.jl - Epigenome/miRNA-wide association: one OLS per outcome column (y ~ covariates),
# returning beta/SE/t/p/95% CI of the first covariate, plus the vectorised lm_series from gen_library.jl.
# Input: CSV with outcomes in columns 1:20 and covariates in 21:22; path from ENV["EWAS_TEST_CSV"]
#        (default <repo>/data/dt_limma_test.csv). A simulated test dataset is kept commented below.
# Packages: DataFrames, Statistics, LinearAlgebra, Distributions, StatsBase, Random, DuckDB (+ gen_library.jl deps)

using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random
using DuckDB  # fix: DuckDB/DBInterface were used but never loaded

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

# Fix: hardcoded absolute path -> env var / repo-relative default
csv_path = replace(get(ENV, "EWAS_TEST_CSV", joinpath(@__DIR__, "..", "data", "dt_limma_test.csv")), '\\' => '/')
con = DBInterface.connect(DuckDB.DB, ":memory:")
dt = DataFrame(DBInterface.execute(con,
           """
           SELECT *
           FROM '$csv_path'
           """));

outc = names(dt)[1:20]
covr = names(dt)[21:22]
# Load library
include("gen_library.jl")

@time fit = lm_series(dt,outc,covr);

# EWAS function 
# Column 2 of X is the first covariate (column 1 is the intercept).
# Fix: p-values are now two-sided (2 * cdf); they were one-sided.
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
            results.pval[i] = 2 * cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs(β[2]/std_coeff[2]))
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
            results.pval[i] = 2 * cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs(β[2]/std_coeff[2]))
            results.ci025[i] = β[2] - quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
            results.ci975[i] = β[2] + quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]            
        end
    end
    return(results)
end
