using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random, MLJ, SIRUS

# Dataset Test for MLJ
rng = MersenneTwister(2090);
x = rand(rng,Normal(0.0,0.1),2000,5);
a = -1.2;
b = vec([1.1 -0.12 0.34 2.13 -1.8]);
mmix = a .+ x * b;
y = rand(rng,MvNormal(mmix,0.5 * I));
dt = DataFrame(hcat(y,x),["y","x1","x2","x3","x4","x5"]);
# dt |> pretty --> nice table visualization
schema(dt);

y, X = unpack(dt, ==(:y); rng=123); # the seed needs to shuffle the observation
# models(matching(X,y)) to check the model that we want to use based on data

# Load model
Tree = @load DecisionTreeRegressor pkg=DecisionTree
tree = Tree()
evaluate(tree, X, y,
        resampling=CV(shuffle=true),
        measures= rsq,
        verbosity=0)

mach = machine(tree, X, y)  
train, test = partition(eachindex(y), 0.7); # 70:30 split
fit!(mach, rows=train);
yhat = MLJ.predict(mach, X[test,:]); 
rmse(yhat,y[test])    
rsquared(yhat,y[test])    

i = info("DecisionTreeRegressor", pkg="DecisionTree")

# Additional interesting topics:
# Stacking 
# Pipeline