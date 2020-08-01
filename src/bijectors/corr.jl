# See stan doc for parametrization method:
# https://mc-stan.org/docs/2_23/reference-manual/correlation-matrix-transform-section.html
# (7/30/2020) their "manageable expression" is wrong...

struct CorrBijector <: Bijector{2} end

(b::CorrBijector)(X::AbstractMatrix{<:Real}) = link_lkj(X)
(b::CorrBijector)(X::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, X)

(ib::Inverse{<:CorrBijector})(Y::AbstractMatrix{<:Real}) = inv_link_lkj(Y)
(ib::Inverse{<:CorrBijector})(Y::AbstractArray{<:AbstractMatrix{<:Real}}) = map(ib, Y)


logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real}) = log_abs_det_jac_lkj(y)
logabsdetjac(b::CorrBijector, X::AbstractMatrix{<:Real}) = - log_abs_det_jac_lkj(b(X))
logabsdetjac(b::CorrBijector, X::AbstractArray{<:AbstractMatrix{<:Real}}) = mapvcat(X) do x
    logabsdetjac(b, x)
end


function log_abs_det_jac_lkj(y)
    # it's defined on inverse mapping
    K = size(y, 1)
    
    z = tanh.(y)
    left = 0
    for i = 1:(K-1), j = (i+1):K
        left += (K-i-1) * log(1 - z[i, j]^2)
    end
    
    right = 0
    for i = 1:(K-1), j = (i+1):K
        right += log(cosh(y[i, j])^2)
    end
    
    return  (0.5 * left - right)
end

function inv_link_w_lkj(y)
    K = size(y, 1)
    ax = 1:K
    ax_roll = [ax[end]; ax[1:end-1]]
    firstrow1 = [ifelse(i==1, 1, 0) for i in 1:K]

    y = y - LowerTriangular(y)

    z = tanh.(y) + I # try to use exp(log(0)) = 0, but AD will stuck on NaN.

    w1 = 0.5 * log.(max.(1 .- z.^2, 1e-6))
    w2 = exp.(cumsum(w1, dims=1))
    w3 = w2[ax_roll, :] .+ firstrow1
    w = w3 .* z

    #=
    # good-looking iteration based code, but AD doesn't like :(
    w = similar(z)
    
    w[1,1] = 1
    for j in 1:K
        w[1, j] = 1
    end

    for i in 2:K
        for j in 1:(i-1)
            w[i, j] = 0
        end
        for j in i:K
            w[i, j] = w[i-1, j] * sqrt(1 - z[i-1, j]^2)
        end
    end

    for i in 1:K
        for j in (i+1):K
            w[i, j] = w[i, j] * z[i, j]
        end
    end
    =#
    
    return w
end

function inv_link_lkj(y)
    w = inv_link_w_lkj(y)
    return w' * w
end

function link_w_lkj(w)
    K = size(w, 1)
    
    #=
    log(z[1, j]) = log(w[1, j])
    log(z[i, j]) = log(w[i, j]) - log(w[i-1, j]) + log(z[i-1, j]) - 0.5 * log(1 - z[i-1, j]^2)
    =#

    # I guess this link function is not required to be differentiable (while it's differentiable for ForwardDiff)
    
    z = zero(w)

    for j=2:K
        z[1, j] = w[1, j]
    end

    for i=2:K, j=(i+1):K
        z[i, j] = w[i, j] / w[i-1, j] * z[i-1, j] / sqrt(1 - z[i-1, j]^2)
    end
    
    y = atanh.(clamp.(z, -1, 1))
    return y
end

function link_lkj(x)
    w = convert(typeof(x), cholesky(x).U) # ? test requires it, such quirk
    return link_w_lkj(w)
end
