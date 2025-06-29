 ; module Domains
using StaticArrays
using LinearAlgebra
export SquareDomain , LDomain , sdf_square , ∇sdf_square , unit_box_normals , unit_box_path , sdf_L , ∇sdf_L , sdf_L_grad , sdf_square_grad;

# Domains
# We define our domains using signed distance functions (SDF) and their gradients.
# The SDF of a unit square centered on \(c \in \RR^{n}\) is given by the \(L^{\infty }\) norm
# \begin{align*}
# \text{sdf}(x) = \|x - c\|_{\infty}
# \end{align*}
# The gradient was calculated analytically and imlpemented as:

 ; 
function sdf_square(x::SVector , r::Float64 , center::SVector)
    return norm(x-center,Inf) .- r
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
end;


# The L shaped Domain can be described by intersecting 2 square SDF centered on \(c = (0,0)\)

# \begin{align*}
# sdf(\Omega_1 \ \Omega_2) &= \max(sdf(\Omega_1) , -sdf(\Omega_2))
# \end{align*}


 ; function sdf_L(x::SVector{2})
    return max(sdf_square(x , 1. , SVector(0,0)) , - sdf_square(x, 1. , SVector(1.,1.)))
end

function ∇sdf_L(x::SVector{2})
    ForwardDiff.gradient(sdf_L , x)
    return
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

 ; end;
