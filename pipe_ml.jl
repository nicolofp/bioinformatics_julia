using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random, MLJ

include("test_s3.jl");
lpheno = S3Path("s3://envbran/methylation/GSE216997_pheno.arrow")
lmirna = S3Path("s3://envbran/methylation/GSE216997_mirna.arrow")

pheno = DataFrame(Arrow.Table(lpheno))
mirna = DataFrame(Arrow.Table(lmirna))

Mmirna = Matrix{Float32}(mirna[:,2:298])
sum(isfinite.(mean(log.(Mmirna), dims = 2)))

