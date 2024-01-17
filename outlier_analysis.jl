using DataFrames, Statistics, LinearAlgebra, OutlierDetection
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

# load the detector
KNN_outlier = @iload KNNDetector pkg=OutlierDetectionNeighbors



# instantiate a detector with default parameters, returning scores
knn = KNN()

# bind the detector to data and learn a model with all data
knn_raw = machine(knn, X) |> fit!

# transform data to raw outlier scores based on the test data; note that there
# is no `predict` defined for raw detectors
transform(knn_raw, rows=test)

# OutlierDetection.jl provides helper functions to normalize the scores,
# for example using min-max scaling based on the training scores
knn_probas = machine(ProbabilisticDetector(knn), X) |> fit!

# predict outlier probabilities based on the test data
predict(knn_probas, rows=test)

# OutlierDetection.jl also provides helper functions to turn scores into classes,
# for example by imposing a threshold based on the training data percentiles
knn_classifier = machine(DeterministicDetector(knn), X) |> fit!

# predict outlier classes based on the test data
predict(knn_classifier, rows=test)