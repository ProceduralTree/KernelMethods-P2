# Preamble

 ; 
module Kernel
using StaticArrays
using KernelAbstractions
using LinearAlgebra;

# Regression Approach
# given \( \hat{X}:=\left\{ x_j \right\}_{j=1}^n \subset\RR ^d\) we aim to find \(u_h(x) \in \mathcal{H}_{k}\) such that it satisfies \eqref{eq:pde} where

# \begin{align}
# \label{eq:approx}
# u_h(x) &= \sum_{j=1}^{n} a_j k(x_j,x)
# \end{align}
# correspondingly we are able to directly compute

# \begin{align*}
# \nabla_x u(x) &= \sum_{j=1}^n a_j \nabla_x  k(x_j ,x) \\
# - \nabla_x \cdot \left( a(x) \nabla_x u(x) \right) &= - \left< \nabla_x a(x) , \nabla_x u(x) \right> - a(x) \Delta_x u(x) \\
# &=  - \sum_{j=1}^{n} a_j \left( \left< \nabla_x a(x) , \nabla_x k(x_j,x)  \right> + a(x) \Delta_x k(x_j,x)\right)
# \end{align*}
# this leads to the following Linear system
# \begin{align}
# \label{eq:1}
# K a &= f & x \in  \Omega
# \end{align}
# where

 ; 
@kernel function system_matrix!(A ,@Const(X), a , ∇a ,k, ∇k, Δk  , sdf , grad_sdf , sdf_beta)
    Iᵢⱼ =  @index(Global , Cartesian)
    @inbounds xᵢ= SVector{2}(view(X, : , Iᵢⱼ[1])) # Essentially X[:,i]
    @inbounds xⱼ= SVector{2}(view(X, : , Iᵢⱼ[2])) # Essentially X[:,j]
    # poisson equation
    @inbounds A[Iᵢⱼ] = -a(xᵢ)*Δk(xᵢ,xⱼ)- ∇a(xᵢ)⋅∇k(xᵢ,xⱼ)
    if abs(sdf(xᵢ)) < 1e-10
        if sdf_beta(xᵢ) < 0
            # Neumann Boundary Condition
            @inbounds nᵢ= grad_sdf(xᵢ)
            @inbounds A[Iᵢⱼ] = a(xᵢ) * (nᵢ ⋅ ∇k(xᵢ , xⱼ))
        else
          # Dirichlet Boundary
          @inbounds A[Iᵢⱼ] =k(xᵢ , xⱼ)
        end
    end
end;

# right hand side

 ; 
@kernel function apply_function_colwise!(B ,@Const(X) , f , g_D , g_N , sdf  , grad_sdf, sdf_beta)
    # boilerplate
    Iᵢ = @index(Global , Cartesian)
    @inbounds xᵢ= SVector{2}(view(X , : , Iᵢ[1]))
    # poisson equation

    @inbounds B[Iᵢ] = f(xᵢ)
    if abs(sdf(xᵢ)) < 1e-10
         if sdf_beta(xᵢ) < 0
             # Neumann Boundary Condition
             @inbounds nᵢ= grad_sdf(xᵢ)
             @inbounds B[Iᵢ] = g_N(xᵢ , nᵢ )
         else
            # Dirichlet Boundary
            @inbounds B[Iᵢ] = g_D(xᵢ)
         end
     end
end;

# Linear Sytem

 ; 
@kernel function linear_matrix!(A ,@Const(X_L) , @Const(X) , k, ∇k , Δk , a , ∇a)
    # boilerplate
    Iᵢⱼ = @index(Global , Cartesian)
    @inbounds xᵢ= SVector{2}(view(X_L , : , Iᵢⱼ[1]))
    @inbounds xⱼ= SVector{2}(view(X , : , Iᵢⱼ[2]))
    # element computation
    @inbounds A[Iᵢⱼ] = ∇a(xᵢ)⋅∇k(xᵢ,xⱼ) -  a(xᵢ)Δk(xⱼ,xᵢ)
    end;

# Dirichlet boundary
# The Dirichlet boundary confitions are dealt with as additional condition in the linear system

 ; 
@kernel function dirichlet_matrix!(A , @Const(X_D) , @Const(X) ,k)
    Iᵢⱼ =  @index(Global , Cartesian)
    @inbounds xᵢ= SVector{2}(view(X_D , : , Iᵢⱼ[1])) # Essentially X[:,1]
    @inbounds xⱼ= SVector{2}(view(X , : , Iᵢⱼ[2]))
    K = k(xᵢ , xⱼ)
    @inbounds A[Iᵢⱼ] = K
end;

# Neumann Boundary


 ; 
@kernel function neumann_matrix!(A , @Const(X_N) , @Const(X) , @Const(N) , a , ∇k )
    Iᵢⱼ =  @index(Global , Cartesian)
    @inbounds xᵢ= SVector{2}(view(X_N , : , Iᵢⱼ[1])) # Essentially X[:,1]
    @inbounds xⱼ= SVector{2}(view(X , : , Iᵢⱼ[2]))
    @inbounds nᵢ= SVector{2}(view(N , : , Iᵢⱼ[1]))
    @inbounds A[Iᵢⱼ] = a(xᵢ) * (nᵢ ⋅ ∇k(xᵢ , xⱼ))
    end;

# Postamble

 ; 
export linear_matrix!
export dirichlet_matrix!
export neumann_matrix!
export apply_function_colwise!
export system_matrix!
end;
