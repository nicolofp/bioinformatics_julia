using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random, MLJ

include("test_s3.jl");
lpheno = S3Path("s3://envbran/methylation/GSE117064_pheno.arrow")
lmirna = S3Path("s3://envbran/methylation/GSE117064_mirna.arrow")

pheno = DataFrame(Arrow.Table(lpheno))
pheno = pheno[pheno.relation .== "",:]
mirna = DataFrame(Arrow.Table(lmirna))

dt = pheno[pheno.source_name_ch1 .== "Non-CVD control",vcat(3,7:9,11:14)]
mirna = mirna[:,vcat("rn",dt.geo_accession)]
mirna.rn = "miRNA" .* string.(1:2565) # Because we have multiple names
Tmirna = permutedims(mirna,1);

# histogram(dt.hba1c)
# Split train-test dataset (now outcome is hba1c)
train, test = partition(collect(eachindex(Tmirna.miRNA1)), 0.8, shuffle=true, rng=111)
Xtrain = MLJ.table(Matrix{Float64}(Tmirna[train,2:2566]))
Xtest  = MLJ.table(Matrix{Float64}(Tmirna[test,2:2566]))
ytrain = Array{Float64}(dt[train,:hba1c])
ytest  = Array{Float64}(dt[test,:hba1c]); 

# Create machine for Lasso Regression 
Standardizer = @load Standardizer pkg=MLJModels
LassoRegressor = LassoRegressor = @load LassoRegressor pkg=MLJLinearModels

# Run the model
# model = Standardizer() |> 
model = LassoRegressor(solver = MLJLinearModels.ProxGrad(max_iter = 10000))
mach = machine(model, Xtrain, ytrain) |> fit! 
evaluate!(mach, resampling = CV(nfolds=10, rng=1234))
report(mach)

coefs, intercept = fitted_params(mach)
lasso_coef = [coefs[i][2] for i in 1:2565]

DT_lasso = DataFrame(name = mirna.rn, coefs = lasso_coef)
DT_lasso[DT_lasso.coefs .> 0,:]

# Compute adj-R2
using Plots
yhat_t =  MLJ.predict(mach, Xtrain)
scatter(ytrain,yhat_t)

# Prediction 
yhat = MLJ.predict(mach, Xtest) 
rms(yhat, ytest)