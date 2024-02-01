using DataFrames, Statistics, LinearAlgebra, OutlierDetection
using Distributions, StatsBase, Random, MLJ

include("test_s3.jl");
lpheno = S3Path("s3://envbran/methylation/GSE117064_pheno.arrow")
lmirna = S3Path("s3://envbran/methylation/GSE117064_mirna.arrow")

pheno = DataFrame(Arrow.Table(lpheno))
pheno = pheno[pheno.relation .== "",:]
mirna = DataFrame(Arrow.Table(lmirna))

dt = pheno[pheno.source_name_ch1 .== "Non-CVD control",vcat(3,7:9,11:14)]
#mirna = mirna[:,vcat("rn",dt.geo_accession)]
mirna.rn = "miRNA" .* string.(1:2565) # Because we have multiple names
Tmirna = permutedims(mirna,1);

# load the detector
KNN = @iload KNNDetector pkg=OutlierDetectionNeighbors

# instantiate a detector with default parameters, returning scores
knn_outlier = KNN()

# bind the detector to data and learn a model with all data
X = MLJ.table(Matrix{Float64}(Tmirna[:,2:2566]))
knn_raw = machine(knn_outlier, X) |> fit!

# transform data to raw outlier scores based on the test data; note that there
# is no `predict` defined for raw detectors
tmp1 = MLJ.transform(knn_raw)#, rows=test)

# OutlierDetection.jl provides helper functions to normalize the scores,
# for example using min-max scaling based on the training scores
knn_probas = machine(ProbabilisticDetector(knn_outlier), X) |> fit!

# predict outlier probabilities based on the test data
tmp = MLJ.predict(knn_probas)#, rows=test)

# Use malahnobis distance and test
Mmirna = Matrix{Float64}(Tmirna[:,2:11])

function mahalanobis_distace(X)
    x = X .- mean(X, dims = 1) 
    B = cov(X)   
    d = sum((x * inv(B)) .* x, dims = 2)
    return(d)     
end

y =mahalanobis_distace(Mmirna)
Chisq(k) 
sum(y .> quantile(Chisq(10), 0.95))

