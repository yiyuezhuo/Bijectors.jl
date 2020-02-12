using LinearAlgebra

struct Conic{B, W, N} <: Bijector{N}
    bs::B
    weights::W
end

conic(bs::B, weights::W) where {N, B<:Tuple{Vararg{<:Bijector{N}}}, W} = Conic{B, W, N}(bs, weights)

closedform(::Inversed{<:Conic}) = false
Bijectors.jacobian(cb::Conic, x::AbstractVector) = sum(i -> cb.weights[i] .* Bijectors.jacobian(cb.bs[i], x), 1:length(cb.bs))

(cb::Conic)(x) = sum(i -> cb.weights[i] .* cb.bs[i](x), 1:length(cb.bs))
function (icb::Inversed{<:Conic})(y)
    cb = icb.orig
    # Using Netwon's method to solve this

    # FIXME: this might be not be in the domain of the bijectors
    x₀ = randn(length(y))

    # Do 10 steps
    x = x₀ - Diagonal(Bijectors.jacobian(cb, x₀)) \ (cb(x₀) - y)
    for i = 1:9
        x = x - Diagonal(Bijectors.jacobian(cb, x)) \ (cb(x) - y)
    end

    return x
end

function Bijectors.logabsdetjac(cb::Conic, x)
    # cb(x) = ∑ᵢ wᵢ bᵢ(x)

    # 1. 𝒥(cb, x) = ∑ᵢ wᵢ 𝒥(bᵢ, x)
    # 2. Assuming monotonicity for all b, |det 𝒥(cb, x)| = det 𝒥(cb, x)
    # 3. log det 𝒥(cb, x) = log det (∑ᵢ wᵢ 𝒥(bᵢ, x))

    # So it works only if we have access to the jacobian of each of the bᵢ's
    Js = [Bijectors.jacobian(b, x) for b in cb.bs]

    return logabsdet(sum(Js .* cb.weights))[1]
end

# TODO: probably don't do this `ForwardDiff` thing...
using ForwardDiff
Bijectors.jacobian(b::Bijector{1}, x::AbstractVector) = ForwardDiff.jacobian(b, x)


