function linreg3(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    (N = length(x)) == length(y) || throw(DimensionMismatch())
    ldiv!(cholesky!(Symmetric([T(N) sum(x); zero(T) sum(abs2, x)], :U)), [sum(y), dot(x, y)])
end

function linreg2(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    X = [ones(length(x)) x]
    ldiv!(cholesky!(Symmetric(X'X, :U)), X'y)
end

# Plot-twist --> I can handle multiple regression in once
function lm_series(dt, out, cov)
    X_tmp = hcat(ones(size(dt,1)), Matrix{Float64}(dt[:,cov]))
    Y_tmp = Matrix{Float64}(dt[:,out])
    β = X_tmp\Y_tmp 
    σ = sqrt.(vec(sum((Y_tmp - X_tmp*β).^2,dims = 1)./(size(X_tmp,1)-size(X_tmp,2))))
    Σ = inv(X_tmp'*X_tmp)
    std_coeff_unscaled = sqrt.(diag(Σ))
    tmp = (beta = β', sigma = σ, S = Σ, 
           df_fitted = (size(X_tmp,1)-size(X_tmp,2)) * ones(size(Y_tmp,2)),
           std_coeff_unscaled = ones(size(Y_tmp,2),size(X_tmp,2)) .* std_coeff_unscaled')
    return tmp
end 

function trigammaInverse(x)
    x[x .> 1e7]  = 1 ./sqrt.(x[x .> 1e7])
    x[x .< 1e-6] = 1 ./ x[x .< 1e-6]
    
    y = 0.5 .+ 1 ./x
    iter = 0 
    dif = 100
    while iter < 100 #niter flexible
        iter += 1
        tri = trigamma.(y)
        dif = tri .* (1 .- tri ./ x) ./ polygamma.(2,y)
        y = y + dif
        if maximum(-dif./y) < 1e-8 break
        end
    end 
    y
end 

function fitFDist(x,df1)
    x[x .< 0] .= 0
    m = median(x)
    x[x .< 1e-5 * m] .= 1e-5 * m
    z = log.(x)
    e = z - digamma.(df1./2) + log.(df1./2)

    emean = mean(e)
    evar = sum((e .- emean).^2)/(length(x)-1)

    evar = evar - mean(trigamma.(df1./2))
    if evar > 0
        df2 = 2*trigammaInverse(evar)
        s20 = exp.(emean .+ digamma.(df2./2)-log.(df2./2))
    else
        df2 = Inf
        s20 = mean(x)
    end
    tmp = (scale=s20,df2=df2)
    return tmp
end         

function squeezeVar_in(var, df, var_prior, df_prior)
    if isfinite(df_prior)
        return (df .* var .+ df_prior .* var_prior) ./ (df .+ df_prior) 
    else var_prior * ones(length(var))
    end
end

function squeezeVar(var, df)
    n = length(var)
    fit = fitFDist(var, df)
    df_prior = fit.df2
    var_post = squeezeVar_in(var, df, fit.scale, df_prior)
    tmp = (df_prior = df_prior, var_prior = fit.scale, var_post = var_post)
    return(tmp)
end

function ebayes(fit; proportion = 0.01,stdev_coef_lim = [0.1,4])
    coefficients = fit.beta
	stdev_unscaled = fit.std_coeff_unscaled
	sigma = fit.sigma
	df_residual = fit.df_fitted

    # Moderated t-stat
    out = squeezeVar(sigma.^2, df_residual)
	s2_prior = out.var_prior
	s2_post = out.var_post
	t_stat = coefficients ./ stdev_unscaled ./ sqrt.(s2_post)
	df_total = df_residual .+ out.df_prior
	df_pooled = sum(df_residual)
	df_total = min(df_total,ones(length(df_total))* df_pooled)
	df_total = df_total
    dist_td = TDist.(df_total)
    p_value = 2 .* cdf.(dist_td, -abs.(t_stat))

    # B-statistic
    # Out of scope for now

    tmp = (df_prior = out.df_prior, s2_prior = s2_prior, s2_post = s2_post,
           tstat = t_stat, df_total = df_total,
           p_value = p_value)
    return tmp
end  

# Put together estimates and shrinked results
function ewas_lm(dt, out, cov)
    fit = lm_series(dt, out, cov)
    tmp = ebayes(fit)
    results = DataFrame(outcome = out,
                        beta = zeros(m),
                        tval = zeros(m),
                        pval = zeros(m),
                        ci025 = zeros(m),
                        ci975 = zeros(m))
    if multithreads == false
        for i in 1:length(out)
            Y_tmp = convert.(Float64, dt[:,out[i]])
            β = X_tmp\Y_tmp
            σ² = sum((Y_tmp - X_tmp*β).^2)/(size(X_tmp,1)-size(X_tmp,2))
            Σ = σ²*inv(X_tmp'*X_tmp)
            std_coeff = sqrt.(diag(Σ))

            results.beta[i] = β[2]
            results.sd[i] = std_coeff[2]
            results.tval[i] = β[2]/std_coeff[2]
            results.pval[i] = cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs(β[2]/std_coeff[2]))
            results.ci025[i] = β[2] - quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
            results.ci975[i] = β[2] + quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]            
        end  
     else
        Threads.@threads for i in 1:length(out)
            Y_tmp = convert.(Float64, dt[:,out[i]])
            β = X_tmp\Y_tmp
            σ² = sum((Y_tmp - X_tmp*β).^2)/(size(X_tmp,1)-size(X_tmp,2))
            Σ = σ²*inv(X_tmp'*X_tmp)
            std_coeff = sqrt.(diag(Σ))

            results.beta[i] = β[2]
            results.sd[i] = std_coeff[2]
            results.tval[i] = β[2]/std_coeff[2]
            results.pval[i] = cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs(β[2]/std_coeff[2]))
            results.ci025[i] = β[2] - quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
            results.ci975[i] = β[2] + quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]            
        end
    end
    return(results)
end