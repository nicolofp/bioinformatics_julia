using DataFrames, Statistics, LinearAlgebra
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


