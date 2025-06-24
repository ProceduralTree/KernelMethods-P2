 ; module PDESolvers
using SparseArrays
using IterativeSolvers

export PDESolver, PDESystem, solve
include("kernel.jl")

using .Kernel
using KernelAbstractions
using LinearAlgebra;

# Solver

# #+RESULTS:


 ; struct PDESystem
    k :: Function
    ∇k :: Function
    Δk :: Function
    a :: Function
    ∇a::Function
    f::Function
    g_D::Function
    g_N::Function
    sdf::Function
    grad_sdf::Function
    sdf_beta::Function
end

struct PDESolver
    S::PDESystem
    X::AbstractMatrix
    α :: AbstractVector
end

function (f::PDESolver)(X)
    dev = get_backend(X)
    print("Backend" , dev)
    K = KernelAbstractions.zeros(dev , Float32, size(X,2)  , size(f.X ,2))
    print("Size of the system Matrix:" , size(K))
    kernel_matrix! = dirichlet_matrix!( dev , 256 , size(K))
    kernel_matrix!(K, X , f.X , f.S.k )
return K * f.α , K
end

function solve(S, X_col)
    dev = get_backend(X_col)
    K = KernelAbstractions.zeros(dev , Float32 , size(X_col , 2) , size(X_col , 2) )
    sys_matrix! = system_matrix!( dev , 256 , size(K))
    sys_matrix!(K ,X_col , S.a , S.∇a , S.k , S.∇k , S.Δk , S.sdf , S.grad_sdf , S.sdf_beta  )
    B = get_boundary(S,X_col)
    α = K \ B
    return (PDESolver(S,X_col ,α) , K)
    end


function get_boundary(
    S,
    X
    )
    dev = get_backend(X)
    B = KernelAbstractions.zeros(dev , Float32 , size(X , 2))
    apply! = apply_function_colwise!(dev , 256 , size(B))
    apply!(B , X , S.f , S.g_D , S.g_N , S.sdf  , S.grad_sdf, S.sdf_beta)
    return B
    end
;

 ; end;
