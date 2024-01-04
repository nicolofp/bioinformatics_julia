using DataFrames, Statistics, LinearAlgebra, Plots
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
Tmirna = permutedims(mirna,1)

# histogram(dt.hba1c)
# Split train-test dataset (now outcome is hba1c)
train, test = partition(collect(eachindex(Tmirna.miRNA1)), 0.8, shuffle=true, rng=111)
Xtrain = MLJ.table(Matrix{Float64}(mirna[train,2:1431]))
Xtest  = MLJ.table(Matrix{Float64}(mirna[test,2:1431]))
ytrain = Array{Float64}(dt[train,:hba1c])
ytest  = Array{Float64}(dt[test,:hba1c]);

# Create machine for PCA 
PCA = @load PCA pkg=MultivariateStats
Standardizer = @load Standardizer pkg=MLJModels

# Create pipeline for dimensionality reduction
model_pca = Standardizer() |> PCA(maxoutdim = 6)
mach_pca = machine(model_pca, Xtrain) |> fit!

# Reconstruct the dataset with PCA
Xtrain_pca = MLJ.transform(mach_pca, Xtrain);
Xtrain_pca = MLJ.table(Xtrain_pca);

# First let's try to LM and the crossvalidation
LinearRegressor = @load LinearRegressor pkg=GLM
model_glm = LinearRegressor()
mach_glm = machine(model_glm, Xtrain_pca, ytrain) |> fit!
fitted_params(mach_glm)
report(mach_glm)

# Let's cross-validate model
evaluate!(mach_glm, resampling = CV(nfolds=10, rng=1234))
report(mach_glm)

# Ensamble example with same ML model
Tree = @load DecisionTreeRegressor pkg=DecisionTree
tree_model = Tree()
mach_tree = machine(tree_model, Xtrain_pca, ytrain) |> fit!
Xtest_pca = MLJ.transform(mach_pca, Xtest);
Xtest_pca = MLJ.table(Xtest_pca);
yhat = MLJ.predict(mach_tree, Xtest_pca) 
rms(yhat, ytest)

evaluate!(mach_tree, resampling=Holdout(fraction_train=0.7, rng=1234),
          measure=rms)

ensemble_model = EnsembleModel(model=tree_model, n=1000);   
ensemble = machine(ensemble_model, Xtrain_pca, ytrain) |> fit!
estimates = evaluate!(ensemble, resampling=CV())

# Staking example with multiple ML models
lm = @load LinearRegressor pkg=GLM
forest = @load RandomForestRegressor pkg=DecisionTree

stack = Stack(;metalearner = lm(),
                resampling=CV(),
                measures=rmse,
                constant = ConstantRegressor(),
                forest = forest(),
                lm = lm())

mach = machine(stack, Xtrain_pca, ytrain) |> fit!
evaluate!(mach; resampling=Holdout(), measure=rmse)

report(mach).cv_report

