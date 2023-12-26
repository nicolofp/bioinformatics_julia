Introduction
================

## Example of test

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
occaecat cupidatat non proident, sunt in culpa qui officia deserunt
mollit anim id est laborum.

## Normalization process

$$KL(\hat{y} || y) = \sum_{c=1}^{M}\hat{y}_c \log{\frac{\hat{y}_c}{y_c}}$$

## Example code

``` julia
using Plots, Statistics, Distributions, DataFrames, DuckDB
#= x = sort(rand(Uniform(0.0,5.0),138))
y = -0.4 .+ 2.926 .* x 
yhat = y + rand(Normal(0.0,1.0),138)
scatter(x,yhat, legend = nothing, title = "Regression line")
plot!(x,y) =#

con = DBInterface.connect(DuckDB.DB, ":memory:")
dt = DataFrame(DBInterface.execute(con,
           """
           SELECT *
           FROM 'C:/Users/nicol/Documents/dt_limma_test.csv'
           """))
show(dt, tf = PrettyTables.tf_markdown)
```

    672×22 DataFrame
     Row | V1        V2         V3        V4         V5        V6        V7        ⋯
         | Float64?  Float64?   Float64?  Float64?   Float64?  Float64?  Float64?  ⋯
    -----|--------------------------------------------------------------------------
       1 | -48.3416  -10.4157   -60.3003  -111.208   103.221   102.298   -107.314  ⋯
       2 | -27.1762   -6.61735  -33.4845   -60.7379   53.5603   53.1157   -59.1148
       3 | -40.9975   -7.96786  -51.6983   -95.9805   92.6236   92.3757   -92.775
       4 | -31.5815   -8.13692  -38.1602   -69.4216   59.6449   58.7894   -67.6502
       5 | -27.1019   -5.8991   -33.9732   -62.3803   57.8498   57.4041   -60.6711 ⋯
       6 | -23.8902   -3.72558  -30.7215   -57.5421   58.1986   58.015    -55.5478
       7 | -45.9441   -9.9532   -57.1808  -105.252    97.4332   96.5771  -101.819
       8 | -32.3188   -6.66026  -40.859    -75.2756   71.5235   71.0518   -72.8946
       9 | -36.2873   -7.51942  -45.1756   -83.6911   78.5308   77.9491   -81.0954 ⋯
      10 | -53.784   -12.3754   -66.6998  -121.956   111.058   110.21    -118.136
      11 | -23.3511   -5.25122  -28.891    -53.0244   47.7208   47.4064   -51.2689
      ⋮  |    ⋮          ⋮         ⋮          ⋮         ⋮         ⋮          ⋮     ⋱
     663 | -38.738    -8.69491  -47.9846   -88.3966   80.955    80.2477   -85.5974
     664 | -45.7994  -11.0053   -56.4102  -103.169    92.2344   91.3191  -100.124  ⋯
     665 | -36.8929   -7.62994  -46.0425   -84.9967   79.6466   79.2011   -82.5019
     666 | -50.4281  -10.036    -63.5305  -117.169   111.369   110.765   -113.398
     667 | -47.1893   -9.08021  -59.9792  -111.396   107.608   107.236   -107.618
     668 | -34.4083   -7.97093  -42.6033   -77.708    70.5073   69.5558   -75.6686 ⋯
     669 | -23.9742   -4.97196  -29.8914   -55.2803   51.9754   51.485    -53.7321
     670 | -29.5614   -7.34937  -36.3877   -66.504    58.3557   57.6202   -64.4813
     671 | -55.5848  -14.3013   -68.0448  -123.637   108.421   107.614   -120.102
     672 | -43.1318   -8.89439  -54.2901  -100.402    95.6209   94.9185   -97.3703 ⋯
                                                     15 columns and 651 rows omitted
