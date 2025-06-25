 ; module PDESolvers
using SparseArrays
using IterativeSolvers

export PDESolver, PDESystem, solve
include("kernel.jl")

using .Kernel
using KernelAbstractions
using LinearAlgebra;

# Solver
# Our implementation Provides some structs for convenience. A ~PDESystem~ that stores the Data functions of a diffusion PDE together with the kernel and its derrivatives \(k , \nabla k , \Delta k\) and information about the Domain of the problem in form if its signed distance function and Its gradient (for normals at the boundary)

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
;


# We provide a  ~PDESolver~ that stores The PDE system, Collocation points \(\hat{X}\) , and the solution vector \(\alpha \) in \(u_h = \sum_{j=0}^n \alpha_j k(x_j , \cdot )\)

 ; struct PDESolver
    S::PDESystem
    X::AbstractMatrix
    α :: AbstractVector
end
;


# ~PDESolver~ Provides methods for evaluation itselve on a test dataset

 ; function (f::PDESolver)(X)
    dev = get_backend(X)
    print("Backend" , dev)
    K = KernelAbstractions.zeros(dev , Float32, size(X,2)  , size(f.X ,2))
    print("Size of the system Matrix:" , size(K))
    km! = kernel_matrix!( dev , 256 , size(K))
    km!(K, X , f.X , f.S.k , f.S.sdf )
return K * f.α , K
end
;



# As well as a method to solve the approximation system and return a instance of ~PDESystem~

 ; function solve(S, X_col)
    dev = get_backend(X_col)
    K = KernelAbstractions.zeros(dev , Float32 , size(X_col , 2) , size(X_col , 2) )
    sys_matrix! = system_matrix!( dev , 256 , size(K))
    sys_matrix!(K ,X_col , S.a , S.∇a , S.k , S.∇k , S.Δk , S.sdf , S.grad_sdf , S.sdf_beta  )
    B = get_boundary(S,X_col)
    α = K \ B
    return (PDESolver(S,X_col ,α) , K)
    end

;

 ; function get_boundary(
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
