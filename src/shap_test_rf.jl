# shap_test_rf.jl - HbA1c in non-CVD controls: standardise + PCA (6 PCs, fit on train) of 2565 miRNAs,
# decision-tree regressor, test RMSE, then Monte Carlo Shapley values (ShapML) on the PC features.
# Input: <repo>/data/GSE117064_pheno.arrow and GSE117064_mirna.arrow (GEO GSE117064 serum miRNA, CVD/stroke;
#        mirna = 2565 miRNA rows x samples, first column "rn"). Folder overridable via ENV["BIOJ_DATA_DIR"].
# Packages: DataFrames, Statistics, LinearAlgebra, Distributions, StatsBase, Random, MLJ, Plots, ShapML, Arrow, MultivariateStats, MLJModels, DecisionTree

using DataFrames, Statistics, LinearAlgebra, Plots
using Distributions, StatsBase, Random, MLJ, ShapML

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

Tree = @load DecisionTreeRegressor pkg=DecisionTree
tree_model = Tree()
mach_tree = machine(tree_model, Xtrain_pca, ytrain) |> fit!
Xtest_pca = MLJ.transform(mach_pca, Xtest);
Xtest_pca = MLJ.table(Xtest_pca);
yhat = MLJ.predict(mach_tree, Xtest_pca) 
rms(yhat, ytest)

evaluate!(mach_tree, resampling=Holdout(fraction_train=0.7, rng=1234),
          measure=rms)

# Create a wrapper function that takes the following positional arguments: (1) a
# trained ML model from any Julia package, (2) a DataFrame of model features. The
# function should return a 1-column DataFrame of predictions--column names do not matter.
function predict_function(model, data)
  data_pred = DataFrame(y_pred = MLJ.predict(model, data))
  return data_pred
end

#------------------------------------------------------------------------------
# ShapML setup.
explain = DataFrame(Xtrain_pca)
explain = explain[1:500,:]

reference = DataFrame(Xtrain_pca) 
sample_size = 60  # Number of Monte Carlo samples.
#------------------------------------------------------------------------------
# Compute stochastic Shapley values.
data_shap = ShapML.shap(explain = explain,
                        reference = reference,
                        model = mach_tree,
                        predict_function = predict_function,
                        sample_size = sample_size,
                        seed = 1
                        )

show(data_shap, allcols = true)
