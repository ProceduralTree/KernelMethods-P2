# Solver

 ; module PDESolvers

export PDESolver, PDESystem, solve
include("kernel.jl")

using .Kernel
using KernelAbstractions
using LinearAlgebra;



# #+RESULTS:


 ; 
struct PDESystem
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
    X_L :: AbstractMatrix
    X_D :: AbstractMatrix
    X_N :: AbstractMatrix
    N :: AbstractMatrix
    α :: AbstractVector
end

function assemble_kernel_matrix(
    S,
    X_L :: AbstractMatrix ,
    X_D :: AbstractMatrix ,
    X_N :: AbstractMatrix ,
    N :: AbstractMatrix
)
    local X = [X_L X_D X_N]
    DOF = size(X,2)
    dev = get_backend(X)
    print("Backend" , dev)
    K = KernelAbstractions.zeros(dev , Float32,DOF ,DOF)
    print("Size of the system Matrix:" , size(K))
    K_linear = @view K[begin:size(X_L , 2) , :]
    K_dirichlet = @view K[size(X_L , 2)+1:end - size(X_N ,2), :]
    K_neumann = @view K[end-size(X_N ,2)+1:end, :]


    cpu_linear! = linear_matrix!( dev , 64 , size(K_linear))
    cpu_dirichlet! = dirichlet_matrix!( dev , 64 , size(K_dirichlet))
    cpu_neumann! = neumann_matrix!( dev , 64 , size(K_neumann))

    cpu_linear!(K_linear  , X_L , X , S.k , S.∇k , S.Δk , S.a , S.∇a)
    @info "Linear"
    cpu_dirichlet!(K_dirichlet  , X_D , X , S.k )
    @info "Dirichlet"
    cpu_neumann!(K_neumann  , X_N , X , N ,S.a, S.∇k)
    @info "Neumann"
    KernelAbstractions.synchronize(dev)
    return K
end
function solve(
    S,
    X_L :: AbstractMatrix ,
    X_D :: AbstractMatrix ,
    X_N :: AbstractMatrix ,
    N :: AbstractMatrix
    )
    K = assemble_kernel_matrix(S, X_L , X_D , X_N , N)
    b = get_boundary(S,X_L , X_D , X_N , N)
    @info "calulating pinv"
    α =  b'*pinv(K)
    @info "calculated pinv"
    return PDESolver(S, X_L , X_D , X_N , N , α' )
    #return b, K

    end
function (f::PDESolver)(X)
    local X_col = [f.X_L f.X_D f.X_N]
    dev = get_backend(X_col)
    print("Backend" , dev)
    K = KernelAbstractions.zeros(dev , Float32, size(X,2)  , size(X_col ,2))
    print("Size of the system Matrix:" , size(K))
    kernel_matrix! = dirichlet_matrix!( dev , 256 , size(K))
    kernel_matrix!(K, X , X_col , f.S.k )
return K * f.α
end

function solve(S, X_col)
    dev = get_backend(X_col)
    K = KernelAbstractions.zeros(dev , Float32 , size(X_col , 2) , size(X_col , 2) )
    sys_matrix! = system_matrix!( dev , 256 , size(K))
    sys_matrix!(K ,X_col , S.a , S.∇a , S.k , S.∇k , S.Δk , S.sdf , S.grad_sdf , S.sdf_beta  )
    end

function get_boundary(
    S,
    X_L::AbstractMatrix ,
    X_D::AbstractMatrix ,
    X_N::AbstractMatrix,
    N::AbstractMatrix
    )
    dev = get_backend(X_L)
    F = KernelAbstractions.zeros(dev , Float32 , size(X_L , 2))
    G_D = KernelAbstractions.zeros(dev , Float32 , size(X_D , 2))
    G_N = KernelAbstractions.zeros(dev , Float32 , size(X_N , 2))
    apply! = apply_function_colwise!(dev , 256)
    apply!(F , X_L , S.f , ndrange=size(F))
    y = [S.f.(eachcol(X_L)); S.g_D.(eachcol(X_D)); S.g_N.(eachcol(X_N) , eachcol(N))]
    end
;

 ; end;
