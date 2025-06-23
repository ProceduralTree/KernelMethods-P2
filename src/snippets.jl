# Distance Matrix Comutation

 ; using KernelAbstractions
using StaticArrays
using Distributed
@kernel function distance_matrix!(A ,@Const(X_1) , @Const(X_2))
    # boilerplate
    Iᵢⱼ = @index(Global , Cartesian)
    @inbounds xᵢ= SVector{3}(view(X_1 , : , Iᵢⱼ[1]))
    @inbounds xⱼ= SVector{3}(view(X_2 , : , Iᵢⱼ[2]))
    # element computation
    @inbounds d = xᵢ - xⱼ
    @inbounds A[Iᵢⱼ] = d' * d
    end



function distM(X₁ ,X₂)
    A = KernelAbstractions.zeros(get_backend(X_1) , Float32 , size(X₁,2) , size(X₂,2))
    dist_kernel! = distance_matrix!(get_backend(A) , 256 , size(A))
    dist_kernel!(A ,X₁ , X₂ )
    KernelAbstractions.synchronize(get_backend(A))
    return A
end

function distK(X_1 , X_2)
norm_1 = sum(X_1.^2 ; dims=1)
norm_2 = sum(X_2.^2 ; dims=1)
distM = -2*(X_1'*X_2) .+ norm_1' .+ norm_2
end;



# #+RESULTS:
# : distK (generic function with 1 method)


 ; using CUDA
X_1 = rand(3,10_000) |> cu
X_2 = rand(3,10_000) |> cu
;

 ; using BenchmarkTools
@benchmark distK(X_1 , X_2);



# #+RESULTS:
# #+begin_example
# BenchmarkTools.Trial: 1012 samples with 1 evaluation per sample.
#  Range (min … max):  49.360 μs … 87.566 ms  ┊ GC (min … max): 0.00% …  0.55%
#  Time  (median):     63.765 μs              ┊ GC (median):    0.00%
#  Time  (mean ± σ):    4.965 ms ± 16.091 ms  ┊ GC (mean ± σ):  1.42% ± 22.26%

#   █                                                        ▁▁
#   █▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▃▁██ ▇
#   49.4 μs      Histogram: log(frequency) by time      56.6 ms <

#  Memory estimate: 15.81 KiB, allocs estimate: 557.
# #+end_example


 ; @benchmark distM(X_1, X_2);

# Kernel Implementation Implementation

 ; using StaticArrays
function rbf_gaussian(r, ::Val{γ}) where γ
    exp(- γ * r)
    end
function rbf_gaussian′(r , ::Val{γ}) where γ
    - γ * exp(- γ * r)
    end
function rbf_gaussian′′(r , ::Val{γ}) where γ
    γ^2 * exp(- γ * r)
    end
function k_gauss(x ::SVector{N}, x̂ ::SVector{N}) where N
    rbf_gaussian(norm(x-x̂)^2 , Val(1.))
    end
function ∇k_gauss(x::SVector{N},x̂::SVector{N}) where N
    rbf_gaussian′(norm(x-x̂)^2 , Val(1.)) * 2*(x-x̂)
    end
function Δk_gauss(x::SVector{N},x̂::SVector{N}) where N
    d = dot(x-x̂,x-x̂)
    4*rbf_gaussian′′(d , Val(1.))  + 4*rbf_gaussian′(d , Val(1.)) * d
    end;



# #+RESULTS:
# : Δk_gauss (generic function with 1 method)


 ; using GLMakie
X = range(-5 , 5 , 100)
Y = range(-5 , 5 , 100)
using LinearAlgebra

fig = Figure()
ax = Axis3(fig[1,1] , aspect=:equal)

gauss(x) = k_gauss(x , [0,0])
z = [gauss([x,y]) for x in X , y in Y]
surface!(ax, X,Y, z)
save("images/gauss-rbf.png",fig );

# PDE System
# such that they satisfy the following system


# \begin{align}
# \label{eq:pde}
# - \nabla  \left( a(x) \nabla u(x) \right) &= f(x) & \text{in} \quad \Omega \\
# u(x) &= g_D(x) & \text{on} \quad  \Gamma_D \\
# \left( a(x) \nabla u(x)  \right) \cdot  \vec{n}(x) &= g_N & \text{on} \quad \Gamma_N
# \end{align}
# where

 ; using StaticArrays
a(x::SVector{2}) = x[1] + 2
∇a(x::SVector{2}) = SVector{2}(1.,0.);



# #+RESULTS:
# : ∇a (generic function with 1 method)


 ; α = 2.
β = 0.5
f(x::SVector{2} , ::Val{α}) where α = - α*norm(x ,2)^(α - 2)*(3x[1] +4) - α*(α -2) * (x[1] + 2) * norm(x,2)^(α - 3)
g_D(x::SVector{2} , ::Val{α}) where α = norm(x,2)^α
g_N(x::SVector{2} , n::SVector{2} , ::Val{α}) where α = α* norm(x,2.)^(α-2.)*(x[1] +2.) * x ⋅ n
f(x) = f(x,Val(α))
g_D(x) = g_D(x,Val(α))
g_N(x) = g_N(x,Val(α));

# Results

 ; using Revise
includet("src/pdesolver.jl")
includet("src/domains.jl");



# #+RESULTS:


 ; using .PDESolvers
using .Domains;



# #+RESULTS:


 ; S = PDESystem(k_gauss , ∇k_gauss , Δk_gauss , a, ∇a , f, g_D ,g_N , sdf_L , sdf_L_grad , sdf_β );



# #+RESULTS:
# : PDESystem(Main.k_gauss, Main.∇k_gauss, Main.Δk_gauss, Main.a, Main.∇a, Main.f, Main.g_D, Main.g_N, Main.Domains.sdf_L, Main.Domains.sdf_L_grad, Main.Domains.sdf_β)






 ; using Random
using CUDA
dev = cu
rng = MersenneTwister(0)
r = 0:0.2:1.99
N = unit_box_normals.(r)
N = reduce(hcat , N) |> dev
X_N = unit_box_path.(r)
X_N = reduce(hcat , X_N)|> dev
X_D = unit_box_path.(2:0.1:4)
X_D = reduce(hcat , X_D) |> dev
X_L = rand(rng , Float64, 2,100) |> dev
;




# #+name: fig:collocation-points

 ; using LaTeXStrings
using Makie
using GLMakie
fig = Figure()
ax = Axis(fig[1,1] , title="Collocation Points")

scatter!(ax,X_L |> Array, label="Data Points")
scatter!(ax,X_D|> Array, label="Dirichlet Points")
scatter!(ax,X_N |> Array, label="Neumann Points")
arrows!(ax,X_N[1,:]|> Array , X_N[2,:] |> Array, N[1,:] |> Array, N[2,:] |> Array, lengthscale=0.1)
axislegend(ax , position=:lt)
save("images/collocation-points.png",fig );



# #+RESULTS: fig:collocation-points
# [[file:images/collocation-points.png]]


 ; using LinearAlgebra
solution = solve(S , X_L , X_D , X_N , N);



# #+RESULTS:
# : julia-async:a974108b-8bfd-4f92-8392-33b6433a07d9

# #+name: fig:solution

 ; using GLMakie
X = range(-2 , 2 , 100)
Y = range(-2 , 2 , 100)
grid = [ [x,y] for x in X , y in Y]
grid = reduce(vcat , grid)
grid = reshape(grid, 2,:)
fig = Figure()
ax = Axis(fig[1,1])
sol = solution(grid)
sol = reshape(sol , size(X,1) , :)
hm = heatmap!(ax , X,Y, sol)
Colorbar(fig[:, end+1], hm)
save("images/solution.png",fig );
