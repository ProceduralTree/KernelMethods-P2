# Domains

 ; module Domains
using StaticArrays
using LinearAlgebra
using ForwardDiff
using Enzyme
export SquareDomain , LDomain , sdf_square , ∇sdf_square , unit_box_normals , unit_box_path , sdf_L , ∇sdf_L , sdf_β , sdf_L_grad , sdf_square_grad;

# Utility

 ; function unit_box_normals(γ::Float64)
    p = SVector{2}(0,0)
    xnormal = SVector{2}(1,0)
    ynormal = SVector{2}(0,1)
    branch = γ % 4.
    if floor(branch) == 0.
        n = -ynormal
    elseif floor(branch) == 1.
        n = xnormal
    elseif floor(branch) == 2.
        n = ynormal
    elseif floor(branch) == 3.
        n = -xnormal
    else
        throw("γ=$γ not in range [0 , 4]")
    end

    return n
end
function unit_box_path(γ::Float64)
    p = SVector{2}(0,0)
    xnormal = SVector{2}(1,0)
    ynormal = SVector{2}(0,1)
    branch = γ % 4.
    if floor(branch) == 0.
        p = branch%1 * xnormal
    elseif floor(branch) == 1.
        p = xnormal +  branch%1 * ynormal
    elseif floor(branch) == 2.
        p = (1-branch%1)*xnormal + ynormal
    elseif floor(branch) == 3.
        p = (1-branch%1) * ynormal
    else
        throw("γ=$γ not in range [0 , 4]")
    end
    return p
end;

 ; 
function sdf_square(x::SVector , r::Float64 , center::SVector)
    return norm(x-center,Inf) .- r
end

function sdf_L(x::SVector{2})
    return max(sdf_square(x , 1. , SVector(0,0)) , - sdf_square(x, 1. , SVector(1.,1.)))
end

function ∇sdf_L(x::SVector{2})
    ForwardDiff.gradient(sdf_L , x)
    return
end

function sdf_β(x::SVector{2})
    return sdf_square(x , 0.2 , SVector(-1.,-1) )
end
function sdf_square_grad(x::SVector{2}, r::Float64, center::SVector{2})
    d = x - center
    if abs(d[1]) > abs(d[2])
        return SVector(sign(d[1]), 0.0)
    elseif abs(d[2]) > abs(d[1])
        return SVector(0.0, sign(d[2]))
    else
        # Subgradient: pick any valid direction; here we average the two
        return normalize(SVector(sign(d[1]), sign(d[2])))
    end
end

function sdf_L_grad(x::SVector{2})
    f1 = sdf_square(x, 1.0, SVector(0.0, 0.0))
    f2 = -sdf_square(x, 1.0, SVector(1.0, 1.0))

    if f1 > f2
        return sdf_square_grad(x, 1.0, SVector(0.0, 0.0))
    elseif f2 > f1
        return -sdf_square_grad(x, 1.0, SVector(1.0, 1.0))  # negative because of the minus
    else
        # Subgradient — average of both directions
        g1 = sdf_square_grad(x, 1.0, SVector(0.0, 0.0))
        g2 = -sdf_square_grad(x, 1.0, SVector(1.0, 1.0))
        return normalize(g1 + g2)
    end
end;

# Postable

 ; end;
