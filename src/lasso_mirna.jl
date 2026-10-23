# lasso_mirna.jl - Lasso regression of systolic blood pressure on 2565 serum miRNAs (non-CVD controls),
# 80/20 train/test split, 10-fold CV on the training set, non-zero coefficients and test RMSE.
# Input: <repo>/data/GSE117064_pheno.arrow and GSE117064_mirna.arrow (GEO GSE117064 serum miRNA, CVD/stroke;
#        mirna = 2565 miRNA rows x samples, first column "rn"). Folder overridable via ENV["BIOJ_DATA_DIR"].
# Packages: DataFrames, Statistics, LinearAlgebra, Distributions, StatsBase, Random, MLJ, Arrow,
#           MLJModels, MLJLinearModels

using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random, MLJ

using Arrow
# Fix: read the local Arrow files instead of include("test_s3.jl") (which loaded AWS credentials just to get Arrow).
# Override the folder with ENV["BIOJ_DATA_DIR"] if the data live elsewhere.
data_dir = get(ENV, "BIOJ_DATA_DIR", joinpath(@__DIR__, "..", "data"))
lpheno = joinpath(data_dir, "GSE117064_pheno.arrow")
lmirna = joinpath(data_dir, "GSE117064_mirna.arrow")

pheno = DataFrame(Arrow.Table(lpheno))
pheno = pheno[pheno.relation .== "",:]
mirna = DataFrame(Arrow.Table(lmirna))

# Fix: select covariates by name; the positional indices vcat(3,7:9,11:14) do not match the Arrow schema
# (they dropped geo_accession and hb-a1c, and :systolic/:hba1c did not exist). Numeric GEO fields are strings.
dt = pheno[pheno.source_name_ch1 .== "Non-CVD control",
           ["geo_accession", "age:ch1", "bmi:ch1", "diastolic bp:ch1", "hb-a1c:ch1", "Sex:ch1", "smoking:ch1", "systolic bp:ch1"]]
rename!(dt, [:geo_accession, :age, :bmi, :diastolic, :hba1c, :sex, :smoking, :systolic])
for c in [:bmi, :diastolic, :hba1c, :systolic]
    dt[!, c] = passmissing(x -> something(tryparse(Float64, x), missing)).(dt[!, c])
end
mirna = mirna[:,vcat("rn",dt.geo_accession)]
mirna.rn = "miRNA" .* string.(1:2565) # Because we have multiple names
Tmirna = permutedims(mirna,1);

# histogram(dt.hba1c)
# Split train-test dataset (now outcome is hba1c)
train, test = partition(collect(eachindex(Tmirna.miRNA1)), 0.8, shuffle=true, rng=111)
#= Xtrain = MLJ.table(Matrix{Float64}(Tmirna[train,2:2566]))
Xtest  = MLJ.table(Matrix{Float64}(Tmirna[test,2:2566]))
ytrain = Array{Float64}(dt[train,:systolic])
ytest  = Array{Float64}(dt[test,:systolic]); =#

X = MLJ.table(Matrix{Float64}(Tmirna[:,2:2566]))
y = Array{Float64}(dt[:,:systolic])
# Fix: Xtest/ytest were used below but only defined in the commented block
Xtest = selectrows(X, test)
ytest = y[test]

# Create machine for Lasso Regression 
Standardizer = @load Standardizer pkg=MLJModels
LassoRegressor = @load LassoRegressor pkg=MLJLinearModels

# Run the model
# model = Standardizer() |> 
model = LassoRegressor(solver = MLJLinearModels.ProxGrad(max_iter = 10000))
mach = machine(model, X, y) #|> fit!
fit!(mach, rows = train)
evaluate!(mach, resampling = CV(nfolds=10, rng=1234), rows = train)  # fix: CV on training rows only (no test leakage)
report(mach)

coefs, intercept = fitted_params(mach)
lasso_coef = [coefs[i][2] for i in 1:2565]

DT_lasso = DataFrame(name = mirna.rn, coefs = lasso_coef)
DT_lasso[DT_lasso.coefs .> 0,:]

# Prediction 
yhat = MLJ.predict(mach, Xtest) 
rms(yhat, ytest)

evaluate!(mach,
          resampling = CV(nfolds = 3, rng = 1234), rows = train,  # fix: seeded, training rows only
          measure = [rsquared])
