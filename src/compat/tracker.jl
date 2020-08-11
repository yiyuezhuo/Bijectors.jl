using .Tracker: Tracker,
                TrackedReal,
                TrackedVector,
                TrackedMatrix,
                TrackedArray,
                TrackedVecOrMat,
                @grad,
                track,
                data,
                param

using Compat: eachcol
using LinearAlgebra

import Base.*

maporbroadcast(f, x::TrackedArray...) = f.(x...)
function maporbroadcast(
    f,
    x1::TrackedArray{T, N},
    x::AbstractArray{<:TrackedReal}...,
) where {T, N}
    return f.(convert(Array{TrackedReal{T}, N}, x1), x...)
end

_eps(::Type{<:TrackedReal{T}}) where {T} = _eps(T)
function Base.minimum(d::LocationScale{<:TrackedReal})
    m = minimum(d.ρ)
    if isfinite(m)
        return d.μ + d.σ * m
    else
        return m
    end
end
function Base.maximum(d::LocationScale{<:TrackedReal})
    m = maximum(d.ρ)
    if isfinite(m)
        return d.μ + d.σ * m
    else
        return m
    end
end

# AD implementations
function jacobian(
    b::Union{<:ADBijector{<:TrackerAD}, Inverse{<:ADBijector{<:TrackerAD}}},
    x::Real
)
    return data(Tracker.gradient(b, x)[1])
end
function jacobian(
    b::Union{<:ADBijector{<:TrackerAD}, Inverse{<:ADBijector{<:TrackerAD}}},
    x::AbstractVector{<:Real}
)
    # We extract `data` so that we don't return a `Tracked` type
    return data(Tracker.jacobian(b, x))
end

# implementations for Shift bijector
function _logabsdetjac_shift(a::TrackedReal, x::Real, ::Val{0})
    return tracker_shift_logabsdetjac(a, x, Val(0))
end
function _logabsdetjac_shift(a::TrackedReal, x::AbstractVector{<:Real}, ::Val{0})
    return tracker_shift_logabsdetjac(a, x, Val(0))
end
function _logabsdetjac_shift(
    a::Union{TrackedReal, TrackedVector{<:Real}},
    x::AbstractVector{<:Real},
    ::Val{1}
)
    return tracker_shift_logabsdetjac(a, x, Val(1))
end
function _logabsdetjac_shift(
    a::Union{TrackedReal, TrackedVector{<:Real}},
    x::AbstractMatrix{<:Real},
    ::Val{1}
)
    return tracker_shift_logabsdetjac(a, x, Val(1))
end
function tracker_shift_logabsdetjac(a, x, ::Val{N}) where {N}
    return param(_logabsdetjac_shift(data(a), data(x), Val(N)))
end

# Log bijector

@grad function logabsdetjac(b::Log{1}, x::AbstractVector)
    return -sum(log, data(x)), Δ -> (nothing, -Δ ./ data(x))
end
@grad function logabsdetjac(b::Log{1}, x::AbstractMatrix)
    return -vec(sum(log, data(x); dims = 1)), Δ -> (nothing, .- Δ' ./ data(x))
end
@grad function logabsdetjac(b::Log{2}, x::AbstractMatrix)
    return -sum(log, data(x)), Δ -> (nothing, -Δ ./ data(x))
end

# implementations for Scale bijector
# Adjoints for 0-dim and 1-dim `Scale` using `Real`
function _logabsdetjac_scale(a::TrackedReal, x::Real, ::Val{0})
    return track(_logabsdetjac_scale, a, data(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::Real, ::Val{0})
    return _logabsdetjac_scale(data(a), data(x), Val(0)), Δ -> (inv(data(a)) .* Δ, nothing, nothing)
end
# Need to treat `AbstractVector` and `AbstractMatrix` separately due to ambiguity errors
function _logabsdetjac_scale(a::TrackedReal, x::AbstractVector, ::Val{0})
    return track(_logabsdetjac_scale, a, data(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{0})
    da = data(a)
    J = fill(inv.(da), length(x))
    return _logabsdetjac_scale(da, data(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
function _logabsdetjac_scale(a::TrackedReal, x::AbstractMatrix, ::Val{0})
    return track(_logabsdetjac_scale, a, data(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{0})
    da = data(a)
    J = fill(size(x, 1) / da, size(x, 2))
    return _logabsdetjac_scale(da, data(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
# adjoints for 1-dim and 2-dim `Scale` using `AbstractVector`
function _logabsdetjac_scale(a::TrackedVector, x::AbstractVector, ::Val{1})
    return track(_logabsdetjac_scale, a, data(x), Val(1))
end
@grad function _logabsdetjac_scale(a::TrackedVector, x::AbstractVector, ::Val{1})
    # ∂ᵢ (∑ⱼ log|aⱼ|) = ∑ⱼ δᵢⱼ ∂ᵢ log|aⱼ|
    #                 = ∂ᵢ log |aᵢ|
    #                 = (1 / aᵢ) ∂ᵢ aᵢ
    #                 = (1 / aᵢ)
    da = data(a)
    J = inv.(da)
    return _logabsdetjac_scale(da, data(x), Val(1)), Δ -> (J .* Δ, nothing, nothing)
end
function _logabsdetjac_scale(a::TrackedVector, x::AbstractMatrix, ::Val{1})
    return track(_logabsdetjac_scale, a, data(x), Val(1))
end
@grad function _logabsdetjac_scale(a::TrackedVector, x::AbstractMatrix, ::Val{1})
    da = data(a)
    Jᵀ = repeat(inv.(da), 1, size(x, 2))
    return _logabsdetjac_scale(da, data(x), Val(1)), Δ -> (Jᵀ * Δ, nothing, nothing)
end
# TODO: implement analytical gradient for scaling a vector using a matrix
# function _logabsdetjac_scale(a::TrackedMatrix, x::AbstractVector, ::Val{1})
#     track(_logabsdetjac_scale, a, data(x), Val{1})
# end
# @grad function _logabsdetjac_scale(a::TrackedMatrix, x::AbstractVector, ::Val{1})
#     throw
# end
# implementations for Stacked bijector
function logabsdetjac(b::Stacked, x::TrackedMatrix{<:Real})
    return map(eachcol(x)) do c
        logabsdetjac(b, c)
    end
end
# TODO: implement custom adjoint since we can exploit block-diagonal nature of `Stacked`
function (sb::Stacked)(x::TrackedMatrix{<:Real})
    return eachcolmaphcat(sb, x)
end
# Simplex adjoints
function _simplex_bijector(X::TrackedVecOrMat, b::SimplexBijector{1})
    return track(_simplex_bijector, X, b)
end
function _simplex_inv_bijector(Y::TrackedVecOrMat, b::SimplexBijector{1})
    return track(_simplex_inv_bijector, Y, b)
end
@grad function _simplex_bijector(X::AbstractVector, b::SimplexBijector{1})
    Xd = data(X)
    return _simplex_bijector(Xd, b), Δ -> (simplex_link_jacobian(Xd)' * Δ, nothing)
end
@grad function _simplex_inv_bijector(Y::AbstractVector, b::SimplexBijector{1})
    Yd = data(Y)
    return _simplex_inv_bijector(Yd, b), Δ -> (simplex_invlink_jacobian(Yd)' * Δ, nothing)
end

@grad function _simplex_bijector(X::AbstractMatrix, b::SimplexBijector{1})
    Xd = data(X)
    return _simplex_bijector(Xd, b), Δ -> begin
        maphcat(eachcol(Xd), eachcol(Δ)) do c1, c2
            simplex_link_jacobian(c1)' * c2
        end, nothing
    end
end
@grad function _simplex_inv_bijector(Y::AbstractMatrix, b::SimplexBijector{1})
    Yd = data(Y)
    return _simplex_inv_bijector(Yd, b), Δ -> begin
        maphcat(eachcol(Yd), eachcol(Δ)) do c1, c2
            simplex_invlink_jacobian(c1)' * c2
        end, nothing
    end
end

replace_diag(::typeof(log), X::TrackedMatrix) = track(replace_diag, log, X)
@grad function replace_diag(::typeof(log), X)
    Xd = data(X)
    f(i, j) = i == j ? log(Xd[i, j]) : Xd[i, j]
    out = f.(1:size(Xd, 1), (1:size(Xd, 2))')
    out, ∇ -> begin
        g(i, j) = i == j ? ∇[i, j]/Xd[i, j] : ∇[i, j]
        return (nothing, g.(1:size(Xd, 1), (1:size(Xd, 2))'))
    end
end

replace_diag(::typeof(exp), X::TrackedMatrix) = track(replace_diag, exp, X)
@grad function replace_diag(::typeof(exp), X)
    Xd = data(X)
    f(i, j) = ifelse(i == j, exp(Xd[i, j]), Xd[i, j])
    out = f.(1:size(Xd, 1), (1:size(Xd, 2))')
    out, ∇ -> begin
        g(i, j) = ifelse(i == j, ∇[i, j]*exp(Xd[i, j]), ∇[i, j])
        return (nothing, g.(1:size(Xd, 1), (1:size(Xd, 2))'))
    end
end

logabsdetjac(b::SimplexBijector{1}, x::TrackedVecOrMat) = track(logabsdetjac, b, x)
@grad function logabsdetjac(b::SimplexBijector{1}, x::AbstractVector)
    xd = data(x)
    return logabsdetjac(b, xd), Δ -> begin
        (nothing, simplex_logabsdetjac_gradient(xd) * Δ)
    end
end
@grad function logabsdetjac(b::SimplexBijector{1}, x::AbstractMatrix)
    xd = data(x)
    return logabsdetjac(b, xd), Δ -> begin
        (nothing, maphcat(eachcol(xd), Δ) do c, g
            simplex_logabsdetjac_gradient(c) * g
        end)
    end
end

for header in [
    (:(u::TrackedArray), :w),
    (:u, :(w::TrackedArray)),
    (:(u::TrackedArray), :(w::TrackedArray)),
]
    @eval begin
        function get_u_hat($(header...))
            if u isa TrackedArray
                T = typeof(u)
            else
                T = typeof(w)
            end
            x = w' * u
            return (u .+ (planar_flow_m(x) - x) .* w ./ sum(abs2, w))::T
        end
    end
end

for header in [
    (:(z::TrackedArray), :w, :b),
    (:z, :(w::TrackedArray), :b),
    (:z, :w, :(b::TrackedReal)),
    (:(z::TrackedArray), :(w::TrackedArray), :b),
    (:(z::TrackedArray), :w, :(b::TrackedReal)),
    (:z, :(w::TrackedArray), :(b::TrackedReal)),
    (:(z::TrackedArray), :(w::TrackedArray), :(b::TrackedReal)),
]
    @eval begin
        function ψ($(header...))
            if z isa AbstractMatrix
                if z isa TrackedMatrix
                    T = typeof(z)
                elseif w isa TrackedVector
                    T = matrixof(typeof(w))
                else
                    T = matrixof(typeof(b))
                end
            else
                if z isa TrackedVector
                    T = typeof(z)
                elseif w isa TrackedVector
                    T = typeof(w)
                else
                    T = vectorof(typeof(b))
                end
            end
            return ((1 .- tanh.(w' * z .+ b).^2) .* w)::T    # for planar flow from eq(11)
        end
    end
end

for header in [
    (:(u::TrackedArray), :w, :b, :(z::AbstractVecOrMat)),
    (:u, :(w::TrackedArray), :b, :(z::AbstractVecOrMat)),
    (:u, :w, :(b::TrackedReal), :(z::AbstractVecOrMat)),
    (:u, :w, :b, :(z::TrackedVecOrMat)),
    (:(u::TrackedArray), :(w::TrackedArray), :b, :(z::AbstractVecOrMat)),
    (:(u::TrackedArray), :w, :(b::TrackedReal), :(z::AbstractVecOrMat)),
    (:(u::TrackedArray), :w, :b, :(z::TrackedVecOrMat)),
    (:u, :(w::TrackedArray), :(b::TrackedReal), :(z::AbstractVecOrMat)),
    (:u, :(w::TrackedArray), :b, :(z::TrackedVecOrMat)),
    (:u, :w, :(b::TrackedArray), :(z::TrackedVecOrMat)),
    (:(u::TrackedArray), :(w::TrackedArray), :(b::TrackedReal), :(z::AbstractVecOrMat)),
    (:(u::TrackedArray), :(w::TrackedArray), :b, :(z::TrackedVecOrMat)),
    (:(u::TrackedArray), :w, :(b::TrackedReal), :(z::TrackedVecOrMat)),
    (:u, :(w::TrackedArray), :(b::TrackedReal), :(z::TrackedVecOrMat)),
    (:(u::TrackedArray), :(w::TrackedArray), :(b::TrackedReal), :(z::TrackedVecOrMat)),
]
    @eval begin
        function _planar_transform($(header...))
            u_hat = get_u_hat(u, w)
            if z isa AbstractVector
                temp = w' * z + b + zero(eltype(u_hat))
                if z isa TrackedVector
                    T = typeof(z)
                elseif u_hat isa TrackedVector
                    T = typeof(u_hat)
                else
                    T = vectorof(typeof(temp))
                end
            else
                temp = w' * z .+ (b + zero(eltype(u_hat)))
                if z isa TrackedMatrix
                    T = typeof(z)
                elseif u_hat isa TrackedVector
                    T = matrixof(typeof(u_hat))
                else
                    T = matrixof(typeof(temp'))
                end
            end
            transformed::T = z .+ u_hat .* tanh.(temp) # from eq(10)
            return (transformed = transformed, u_hat = u_hat)
        end
    end
end

for header in [
    (:(α_::TrackedReal), :β, :z_0, :(z::AbstractVector)),
    (:α_, :(β::TrackedReal), :z_0, :(z::AbstractVector)),
    (:α_, :β, :(z_0::TrackedVector), :(z::AbstractVector)),
    (:α_, :β, :z_0, :(z::TrackedVector)),
    (:(α_::TrackedReal), :(β::TrackedReal), :z_0, :(z::AbstractVector)),
    (:(α_::TrackedReal), :β, :(z_0::TrackedVector), :(z::AbstractVector)),
    (:(α_::TrackedReal), :β, :z_0, :(z::TrackedVecOrMat)),
    (:(α_::TrackedReal), :(β::TrackedReal), :(z_0::TrackedVector), :(z::AbstractVector)),
    (:(α_::TrackedReal), :(β::TrackedReal), :z_0, :(z::TrackedVector)),
    (:(α_::TrackedReal), :β, :(z_0::TrackedVector), :(z::TrackedVector)),
    (:α_, :(β::TrackedReal), :(z_0::TrackedVector), :(z::TrackedVector)),
    (:(α_::TrackedReal), :(β::TrackedReal), :(z_0::TrackedVector), :(z::TrackedVector)),
]
    @eval begin
        function _radial_transform($(header...))
            α = softplus(α_)            # from A.2
            β_hat = -α + softplus(β)    # from A.2
            if β_hat isa TrackedReal
                TV = vectorof(typeof(β_hat))
                T = vectorof(typeof(β_hat))
            elseif z_0 isa TrackedVector
                TV = typeof(z_0)
                T = typeof(z_0)
            else
                T = TV = typeof(z)
            end
            Tr = promote_type(eltype(z), eltype(z_0))
            r::Tr = norm((z .- z_0)::TV)
            transformed::T = z .+ β_hat ./ (α .+ r') .* (z .- z_0)   # from eq(14)
            return (transformed = transformed, α = α, β_hat = β_hat, r = r)
        end
    end
end

for header in [
    (:(α_::TrackedReal), :β, :z_0, :(z::AbstractMatrix)),
    (:α_, :(β::TrackedReal), :z_0, :(z::AbstractMatrix)),
    (:α_, :β, :(z_0::TrackedVector), :(z::AbstractMatrix)),
    (:α_, :β, :z_0, :(z::TrackedMatrix)),
    (:(α_::TrackedReal), :(β::TrackedReal), :z_0, :(z::AbstractMatrix)),
    (:(α_::TrackedReal), :β, :(z_0::TrackedVector), :(z::AbstractMatrix)),
    (:(α_::TrackedReal), :β, :z_0, :(z::TrackedMatrix)),
    (:(α_::TrackedReal), :(β::TrackedReal), :(z_0::TrackedVector), :(z::AbstractMatrix)),
    (:(α_::TrackedReal), :(β::TrackedReal), :z_0, :(z::TrackedMatrix)),
    (:(α_::TrackedReal), :β, :(z_0::TrackedVector), :(z::TrackedMatrix)),
    (:α_, :(β::TrackedReal), :(z_0::TrackedVector), :(z::TrackedMatrix)),
    (:(α_::TrackedReal), :(β::TrackedReal), :(z_0::TrackedVector), :(z::TrackedMatrix)),
]
    @eval begin
        function _radial_transform($(header...))
            α = softplus(α_)            # from A.2
            β_hat = -α + softplus(β)    # from A.2
            if β_hat isa TrackedReal
                TV = vectorof(typeof(β_hat))
                T = matrixof(TV)
            elseif z_0 isa TrackedVector
                TV = typeof(z_0)
                T = matrixof(TV)
            else
                T = typeof(z)
                TV = vectorof(T)
            end
            r::TV = eachcolnorm(z .- z_0)
            transformed::T = z .+ β_hat ./ (α .+ r') .* (z .- z_0)   # from eq(14)
            return (transformed = transformed, α = α, β_hat = β_hat, r = r)
        end
    end
end
eachcolnorm(X) = map(norm, eachcol(X))
eachcolnorm(X::TrackedMatrix) = track(eachcolnorm, X)
@grad function eachcolnorm(X)
    Xd = data(X)
    y = map(norm, eachcol(Xd))
    y, Δ -> begin
        (Xd .* (Δ ./ y)',)
    end
end

function matrixof(::Type{TrackedArray{T, 1, Vector{T}}}) where {T <: Real}
    return TrackedArray{T, 2, Matrix{T}}
end
function matrixof(::Type{TrackedReal{T}}) where {T <: Real}
    return TrackedArray{T, 2, Matrix{T}}
end
function vectorof(::Type{TrackedArray{T, 2, Matrix{T}}}) where {T <: Real}
    return TrackedArray{T, 1, Vector{T}}
end
function vectorof(::Type{TrackedReal{T}}) where {T <: Real}
    return TrackedArray{T, 1, Vector{T}}
end

(b::Exp{0})(x::TrackedVector) = exp.(x)::vectorof(float(eltype(x)))
(b::Exp{1})(x::TrackedVector) = exp.(x)::vectorof(float(eltype(x)))
(b::Exp{1})(x::TrackedMatrix) = exp.(x)::matrixof(float(eltype(x)))
(b::Exp{2})(x::TrackedMatrix) = exp.(x)::matrixof(float(eltype(x)))

(b::Log{0})(x::TrackedVector) = log.(x)::vectorof(float(eltype(x)))
(b::Log{1})(x::TrackedVector) = log.(x)::vectorof(float(eltype(x)))
(b::Log{1})(x::TrackedMatrix) = log.(x)::matrixof(float(eltype(x)))
(b::Log{2})(x::TrackedMatrix) = log.(x)::matrixof(float(eltype(x)))

logabsdetjac(b::Log{0}, x::TrackedVector) = .-log.(x)::vectorof(float(eltype(x)))
logabsdetjac(b::Log{1}, x::TrackedMatrix) = - vec(sum(log.(x); dims = 1))

getpd(X::TrackedMatrix) = track(getpd, X)
@grad function getpd(X::AbstractMatrix)
    Xd = data(X)
    return LowerTriangular(Xd) * LowerTriangular(Xd)', Δ -> begin
        Xl = LowerTriangular(Xd)
        return (LowerTriangular(Δ' * Xl + Δ * Xl),)
    end
end

lower(A::TrackedMatrix) = track(lower, A)
@grad function lower(A::AbstractMatrix)
    Ad = data(A)
    return lower(Ad), Δ -> (lower(Δ),)
end

inv_link_w_lkj(y::TrackedMatrix) = track(inv_link_w_lkj, y)
@grad function inv_link_w_lkj(y_tracked)
    y = data(y_tracked)

    K = size(y, 1)

    z = tanh.(y)
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

    w1 = copy(w) # cache result

    for i in 1:K
        for j in (i+1):K
            w[i, j] = w[i, j] * z[i, j]
        end
    end

    return w, Δw -> begin
        Δz = zeros(size(Δw))
        Δw1 = zeros(size(Δw))
        for i in 1:K, j in (i+1):K
            Δw1[i,j] = Δw[i,j] * z[i,j]
            Δz[i,j] = Δw[i,j] * w1[i,j]
        end
        for i in 1:K
            Δw1[i,i] = Δw[i,i]
        end

        for j=2:K, i=j:-1:2
            tz = sqrt(1 - z[i-1, j]^2)
            Δw1[i-1, j] += Δw1[i, j] * tz
            Δz[i-1, j] += Δw1[i, j] * w1[i-1, j] * 0.5 / tz * (-2 * z[i-1, j])
        end

        Δy = Δz .* (1 ./ cosh.(y).^2)
        return (Δy,)
    end
end

link_w_lkj(w::TrackedMatrix) = track(link_w_lkj, w)
@grad function link_w_lkj(w_tracked)
    w = data(w_tracked)

    K = size(w, 1)
    z = zero(w)
    
    for j=2:K
        z[1, j] = w[1, j]
    end

    #=
    for i=2:K, j=(i+1):K
        z[i, j] = w[i, j] / w[i-1, j] * z[i-1, j] / sqrt(1 - z[i-1, j]^2)
    end
    =#
    for i=2:K, j=(i+1):K
        p = w[i, j]
        for ip in 1:(i-1)
            p *= 1 / sqrt(1-z[ip, j]^2)
        end
        z[i,j] = p
    end
    
    y = atanh.(z)

    return y, Δy -> begin
        zt0 = 1 ./ (1 .- z.^2)
        zt = sqrt.(zt0)
        Δz = Δy .* zt0
        Δw = zeros(size(Δy))
        
        for j=2:K, i=(j-1):-1:2
            pd = prod(zt[1:i-1,j])
            Δw[i,j] += Δz[i,j] * pd
            for ip in 1:(i-1)
                Δw[ip, j] += Δz[i,j] * w[i,j] * pd / (1-z[ip,j]^2) * z[ip,j]
            end
        end
        for j=2:K
            Δw[1, j] += Δz[1, j]
        end

        (Δw,)
    end
    
    #=
    return y, Δy -> begin
        Δz = Δy .* (1 ./ (1. .- z.^2))
        Δw = zeros(size(Δz))
        for j=2:K, i=(j-1):-1:2
            tz = sqrt(1 - z[i-1, j]^2)
            Δw[i,j] += Δz[i,j] / w[i-1,j] * z[i-1, j] / tz
            Δw[i-1,j] += Δz[i,j] * w[i,j] * z[i-1, j] / tz * (-1 / w[i-1, j]^2)
            Δz[i-1,j] += Δz[i,j] * w[i,j] / w[i-1, j] * ((tz - z[i-1,j] * 0.5 / tz * (-2*z[i-1,j])) / tz^2)
        end
        
        for j=2:K
            Δw[1, j] += Δz[1, j]
        end
        
        return (Δw,)
    end
    =#
end

upper1(AT::TrackedMatrix, A::TrackedMatrix) = track(upper1, AT, A)
@grad function upper1(AT_tracked, A_tracked)
    AT = data(AT_tracked)
    A = data(A_tracked)
    return upper1(AT, A), Δ -> (nothing, upper1(AT, Δ))
end

# Workaround for Tracker ambiguous bug. See: https://github.com/FluxML/Tracker.jl/issues/74
# (*)(X::Diagonal, Y::TrackedArray{T,2,A} where A where T) = collect(X) * Y
x::Diagonal * y::TrackedMatrix  = track(*, x, y)
x::TrackedMatrix * y::Diagonal  = track(*, x, y)

function LinearAlgebra.isposdef(w_tracked::TrackedMatrix)
    # w = data(w_tracked)
    w = w_tracked.data
    return LinearAlgebra.isposdef(w)
end

function LinearAlgebra.isposdef(w_tracked::Symmetric{<:Any, <:TrackedMatrix})
    # w = data(w_tracked)
    w = w_tracked.data
    return LinearAlgebra.isposdef(w)
end
