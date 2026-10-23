# stacking.jl - HbA1c in non-CVD controls: standardise + PCA (6 PCs, fit on train) of 2565 miRNAs, then
# compare GLM linear regression, a decision tree, a bagged tree ensemble and a Stack (constant/forest/lm).
# Input: <repo>/data/GSE117064_pheno.arrow and GSE117064_mirna.arrow (GEO GSE117064 serum miRNA, CVD/stroke;
#        mirna = 2565 miRNA rows x samples, first column "rn"). Folder overridable via ENV["BIOJ_DATA_DIR"].
# Packages: DataFrames, Statistics, LinearAlgebra, Distributions, StatsBase, Random, MLJ, Plots, Arrow, MultivariateStats, MLJModels, GLM (MLJGLMInterface), DecisionTree

using DataFrames, Statistics, LinearAlgebra, Plots
using Distributions, StatsBase, Random, MLJ

using Arrow
# Fix: read the local Arrow files instead of include("test_s3.jl") (which loaded AWS credentials just to get Arrow).
# Override the folder with ENV["BIOJ_DATA_DIR"] if the data live elsewhere.
data_dir = get(ENV, "BIOJ_DATA_DIR", joinpath(@__DIR__, "..", "data"))
lpheno = joinpath(data_dir, "GSE117064_pheno.arrow")
lmirna = joinpath(data_dir, "GSE117064_mirna.arrow")
Random.seed!(1234)  # fix: seed global RNG (tree/forest/ensemble models default to it)

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
Tmirna = permutedims(mirna,1)

# histogram(dt.hba1c)
# Split train-test dataset (now outcome is hba1c)
train, test = partition(collect(eachindex(Tmirna.miRNA1)), 0.8, shuffle=true, rng=111)
# Fix: rows must be samples (Tmirna, 2565 miRNA columns); the old code indexed the miRNA x sample matrix
Xtrain = MLJ.table(Matrix{Float64}(Tmirna[train,2:2566]))
Xtest  = MLJ.table(Matrix{Float64}(Tmirna[test,2:2566]))
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

# Ensamble model          
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

