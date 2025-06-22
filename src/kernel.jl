# Preamble

 ; 
module Kernel
using StaticArrays
using KernelAbstractions
using LinearAlgebra
using ForwardDiff;

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
    if isnan(K)
        @print(Iᵢⱼ , "\n")
        @print(xᵢ , "\n")
        @print(xⱼ , "\n")
        end
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

# right hand side

 ; 
@kernel function apply_function_colwise!(A ,@Const(X) , f::Function)
    # boilerplate
    Iᵢ = @index(Global , Cartesian)
    @inbounds xᵢ= SVector{2}(view(X , : , Iᵢ[1]))
    # element computation
    @inbounds A[Iᵢ] = f(xᵢ)
    end;

# Combined System


 ; 
@kernel function system_matrix!(A ,@Const(X), a , ∇a ,k, ∇k, Δk  , sdf , grad_sdf , sdf_beta)
    Iᵢⱼ =  @index(Global , Cartesian)
    @inbounds xᵢ= SVector{2}(view(X, : , Iᵢⱼ[1])) # Essentially X[:,1]
    @inbounds xⱼ= SVector{2}(view(X , : , Iᵢⱼ[2]))
    if sdf(xᵢ) < 1e-10
        if sdf_beta(xᵢ) < 0
            @inbounds nᵢ= grad_sdf(xᵢ)
            @inbounds A[Iᵢⱼ] = a(xᵢ) * (nᵢ ⋅ ∇k(xᵢ , xⱼ))
        else
           @inbounds A[Iᵢⱼ] =k(xᵢ , xⱼ)
        end
    else
        @inbounds A[Iᵢⱼ] = ∇a(xᵢ)⋅∇k(xᵢ,xⱼ) -  a(xᵢ)*Δk(xⱼ,xᵢ)
    end
    end;

# Postable

 ; 
export linear_matrix!
export dirichlet_matrix!
export neumann_matrix!
export apply_function_colwise!
export system_matrix!
end;
