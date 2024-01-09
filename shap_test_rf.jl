using DataFrames, Statistics, LinearAlgebra, Plots
using Distributions, StatsBase, Random, MLJ, ShapML

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