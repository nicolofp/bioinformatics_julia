using DataFrames, Statistics, LinearAlgebra 
using Distributions, StatsBase, Random

# Dataset Test
rng = MersenneTwister(2090);
x = rand(rng,Normal(0.0,0.1),2000,5)
a = -1.2
b = vec([1.1 -0.12 0.34 2.13 -1.8])
b2 = vec([2.1 -1.12 0.94 -1.13 -0.38])
mmix = a .+ x * b
mmix2 = a .+ x * b2
y1 = rand(rng,MvNormal(mmix,0.5 * I))
y2 = rand(rng,MvNormal(mmix,0.5 * I))
y3 = rand(rng,MvNormal(mmix2,0.5 * I))
y4 = rand(rng,MvNormal(mmix,0.5 * I))
y = hcat(y1,y2,y3,y4)
dt = DataFrame(hcat(y,x), ["y1","y2","y3","y4","x1","x2","x3","x4","x5"])
out = ["y1","y2","y3","y4"];
cov = ["x1","x2","x3","x4","x5"];

ewas_lm(dt,out,cov,true)

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