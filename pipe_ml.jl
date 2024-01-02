using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random, MLJ

include("test_s3.jl");
lpheno = S3Path("s3://envbran/methylation/GSE117064_pheno.arrow")
lmirna = S3Path("s3://envbran/methylation/GSE117064_mirna.arrow")

pheno = DataFrame(Arrow.Table(lpheno))
pheno = pheno[pheno.relation .== "",:]
mirna = DataFrame(Arrow.Table(lmirna))

rn_ncvd  = parse.(Int32,pheno[ismissing.(pheno.pairing_rn) .== 0,:pairing_rn])
rn_cvd = pheno[ismissing.(pheno.pairing_rn) .== 0,:rn]
pheno = pheno[vcat(rn_ncvd,rn_cvd),:]
names(pheno)
mirna = mirna[:,vcat("rn",pheno.geo_accession)]

# Check mean and variance 
std.(eachrow(mirna[:,2:347]))
mean.(eachrow(mirna[:,2:347]))

# I should transform everything in Matrix format 
pheno.diagnosis = parse.(Int64,pheno.diagnosis)
pheno.sex = ifelse.(pheno.sex .== "Male",1,0)
Mmirna = Matrix(Matrix{Float64}(mirna[:,2:347])')
Mpheno = Matrix(pheno[:,[:diagnosis, :age, :sex]])
mirna.rn = "miRNA" .* string.(1:2565)
Tmirna = permutedims(mirna,1)
Tmirna.y = coerce(pheno.source_name_ch1, Multiclass)  

# Create machine for PCA 
PCA = @load PCA pkg=MultivariateStats
Standardizer = @load Standardizer pkg=MLJModels
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree

train, test = partition(collect(eachindex(Tmirna.y)), 0.7, shuffle=true, rng=111)
ytrain = Tmirna.y[train]
ytest = Tmirna.y[test]
Xtrain = MLJ.table(Mmirna[train,:])
Xtest = MLJ.table(Mmirna[test,:])
model_pca = Standardizer() |> PCA(maxoutdim = 6) |> RandomForestClassifier()
#model_rf = 
mach = machine(model_pca, Xtrain, ytrain) |> fit!
Xtrain_pca = MLJ.transform(mach,Xtrain)

mach_rf = machine(model_rf, Xtrain_pca, ytrain) |> fit!
yhat = MLJ.predict_mode(mach, Xtest)
confusion_matrix(yhat, ytest)
accuracy(yhat, ytest)
# ex_var = report(mach).pca.principalvars
# sum((ex_var.^2/sum(ex_var.^2))[1:6])
# fitted_params(mach)
# report(mach).loadings