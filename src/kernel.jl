# Preamble

 ; 
module Kernel
using StaticArrays
using KernelAbstractions
using LinearAlgebra;

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

# Postable

 ; 
export linear_matrix!
export dirichlet_matrix!
export neumann_matrix!
end;
