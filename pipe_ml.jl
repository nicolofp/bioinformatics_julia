using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random, MLJ

include("test_s3.jl");
lpheno = S3Path("s3://envbran/methylation/GSE117064_pheno.arrow")
lmirna = S3Path("s3://envbran/methylation/GSE117064_mirna.arrow")

pheno = DataFrame(Arrow.Table(lpheno));
mirna = DataFrame(Arrow.Table(lmirna));

pheno = pheno[pheno.class_label .== 1,["geo_accession", "source_name_ch1", "age:ch1", "Sex:ch1"]];
rename!(pheno,[:geo_accession,:diagnosis,:age,:sex]);
mirna = mirna[:,vcat("rn",pheno.geo_accession)];

# I should transform everything in Matrix format 
pheno.sex = ifelse.(pheno.sex .== "Male",1,0)
Mmirna = Matrix(Matrix{Float64}(mirna[:,2:347])')
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
model2 = RandomForestClassifier()
mach = machine(model, Xtrain) |> fit! 
Xtrain_pca = MLJ.transform(mach, Xtrain)


mach2 = machine(model2, Xtrain_pca, ytrain) |> fit!

# Cross-validation
evaluate!(mach2, resampling = CV(nfolds=10, rng=1234), measure = accuracy)

# Summary 
Xtest_pca = MLJ.transform(mach, Xtest)
yhat = MLJ.predict_mode(mach2, Xtest_pca)
confusion_matrix(yhat, ytest)
accuracy(yhat, ytest)

# ex_var = report(mach).pca.principalvars
# sum((ex_var.^2/sum(ex_var.^2))[1:6])
# fitted_params(mach)
# report(mach).loadings

