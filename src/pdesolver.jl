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
    K = zeros(DOF ,DOF)
    K_linear = @view K[begin:size(X_L , 2) , :]
    K_dirichlet = @view K[size(X_L , 2)+1:end - size(X_N ,2), :]
    K_neumann = @view K[end-size(X_N ,2)+1:end, :]


    cpu_linear! = linear_matrix!( CPU() , 64 , size(K_linear))
    cpu_dirichlet! = dirichlet_matrix!( CPU() , 64 , size(K_dirichlet))
    cpu_neumann! = neumann_matrix!( CPU() , 64 , size(K_neumann))

    cpu_linear!(K_linear  , X_L , X , S.k , S.∇k , S.Δk , S.a , S.∇a)
    cpu_dirichlet!(K_dirichlet  , X_D , X , S.k )
    cpu_neumann!(K_neumann  , X_N , X , N ,S.a, S.∇k)
    KernelAbstractions.synchronize(get_backend(K))
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
    α =  b'*pinv(K)
    return PDESolver(S, X_L , X_D , X_N , N , α' )
    #return b, K

    end
function (f::PDESolver)(X)
    local X_col = [f.X_L f.X_D f.X_N]
    K = zeros(size(X,2)  , size(X_col ,2))
    kernel_matrix! = dirichlet_matrix!( CPU() , 64 , size(K))
    kernel_matrix!(K, X , X_col , f.S.k )
return K * f.α
end

function get_boundary(
    S,
    X_L::AbstractMatrix ,
    X_D::AbstractMatrix ,
    X_N::AbstractMatrix,
    N::AbstractMatrix
    )
    y = [S.f.(eachcol(X_L)); S.g_D.(eachcol(X_D)); S.g_N.(eachcol(X_N) , eachcol(N))]
    end
;

 ; end;
