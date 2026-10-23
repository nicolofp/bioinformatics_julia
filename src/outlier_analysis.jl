# outlier_analysis.jl - Sample-level outlier detection on the miRNA matrix: KNN detector scores/probabilities
# (OutlierDetection.jl) and squared Mahalanobis distance on 10 miRNAs vs chi-square(10) 95% cutoff.
# Input: <repo>/data/GSE117064_pheno.arrow and GSE117064_mirna.arrow (GEO GSE117064 serum miRNA, CVD/stroke;
#        mirna = 2565 miRNA rows x samples, first column "rn"). Folder overridable via ENV["BIOJ_DATA_DIR"].
# Packages: DataFrames, Statistics, LinearAlgebra, OutlierDetection, OutlierDetectionNeighbors,
#           Distributions, StatsBase, Random, MLJ, Arrow

using DataFrames, Statistics, LinearAlgebra, OutlierDetection
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

# Squared Mahalanobis distance of each row from the column means (~ chi-square(ncol) under normality)
function mahalanobis_distace(X)
    x = X .- mean(X, dims = 1) 
    B = cov(X)   
    d = sum((x * inv(B)) .* x, dims = 2)
    return(d)     
end

y = mahalanobis_distace(Mmirna)
# Chisq(k)  # fix: `k` was undefined and aborted the script
# df Chisq is number of columns (with high p?)
sum(y .> quantile(Chisq(10), 0.95))

