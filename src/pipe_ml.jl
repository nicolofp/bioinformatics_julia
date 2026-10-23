# pipe_ml.jl - CVD patient vs non-CVD control classification (class_label == 1 subset, 346 samples):
# standardise + PCA (6 PCs, fit on the 70% train split) then random forest; CV accuracy and test confusion matrix.
# Input: <repo>/data/GSE117064_pheno.arrow and GSE117064_mirna.arrow (GEO GSE117064 serum miRNA, CVD/stroke;
#        mirna = 2565 miRNA rows x samples, first column "rn"). Folder overridable via ENV["BIOJ_DATA_DIR"].
# Packages: DataFrames, Statistics, LinearAlgebra, Distributions, StatsBase, Random, MLJ, Arrow,
#           MultivariateStats, MLJModels, DecisionTree

using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random, MLJ

using Arrow
# Fix: read the local Arrow files instead of include("test_s3.jl") (which loaded AWS credentials just to get Arrow).
# Override the folder with ENV["BIOJ_DATA_DIR"] if the data live elsewhere.
data_dir = get(ENV, "BIOJ_DATA_DIR", joinpath(@__DIR__, "..", "data"))
lpheno = joinpath(data_dir, "GSE117064_pheno.arrow")
lmirna = joinpath(data_dir, "GSE117064_mirna.arrow")

pheno = DataFrame(Arrow.Table(lpheno));
mirna = DataFrame(Arrow.Table(lmirna));

pheno = pheno[pheno.class_label .== 1,["geo_accession", "source_name_ch1", "age:ch1", "Sex:ch1"]];
rename!(pheno,[:geo_accession,:diagnosis,:age,:sex]);
mirna = mirna[:,vcat("rn",pheno.geo_accession)];

# I should transform everything in Matrix format 
pheno.sex = ifelse.(pheno.sex .== "Male",1,0)
Mmirna = Matrix(Matrix{Float64}(mirna[:,2:347])')  # samples x miRNAs, same order as pheno
Mpheno = Matrix(pheno[:,[:diagnosis, :age, :sex]])
mirna.rn = "miRNA" .* string.(1:2565)
Tmirna = permutedims(mirna,1)
Tmirna.y = coerce(pheno.diagnosis, OrderedFactor);  

# Create machine for PCA 
PCA = @load PCA pkg=MultivariateStats
Standardizer = @load Standardizer pkg=MLJModels
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree


train, test = partition(collect(eachindex(Tmirna.y)), 0.7, shuffle=true, rng=111)
ytrain = Tmirna.y[train]
ytest = Tmirna.y[test]
Xtrain = MLJ.table(Mmirna[train,:])
Xtest = MLJ.table(Mmirna[test,:])
model = Standardizer() |> PCA(maxoutdim = 6) #|> RandomForestClassifier()
model2 = RandomForestClassifier(rng = 1234)  # fix: seeded forest (reported accuracy)
mach = machine(model, Xtrain) |> fit! 
Xtrain_pca = MLJ.transform(mach, Xtrain);

mach2 = machine(model2, Xtrain_pca, ytrain) |> fit!

# Cross-validation
# Fix: accuracy needs point predictions from a probabilistic classifier -> operation = predict_mode
evaluate!(mach2, resampling = CV(nfolds=10, rng=1234), measure = accuracy, operation = predict_mode)

# Summary 
Xtest_pca = MLJ.transform(mach, Xtest);
yhat = MLJ.predict_mode(mach2, Xtest_pca);
confusion_matrix(yhat, ytest)
accuracy(yhat, ytest)
