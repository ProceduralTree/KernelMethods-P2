# Distance Matrix Comutation :noexport:

 ; 
using KernelAbstractions
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

 ; using CUDA
using OpenCL
dev = CUDA.functional() ? cu : Array
#dev = CLArray
X_1 = rand(3,10_00) |> dev
X_2 = rand(3,10_00) |> dev
;

 ; using BenchmarkTools
@benchmark distK(X_1 , X_2);



# #+RESULTS:
# #+begin_example
# BenchmarkTools.Trial: 5 samples with 1 evaluation per sample.
#  Range (min … max):  939.480 ms …    1.252 s  ┊ GC (min … max):  0.36% … 25.33%
#  Time  (median):        1.205 s               ┊ GC (median):    22.04%
#  Time  (mean ± σ):      1.124 s ± 146.145 ms  ┊ GC (mean ± σ):  16.61% ± 11.64%

#   █         █                                        █    █   █
#   █▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁█▁▁▁█ ▁
#   939 ms           Histogram: frequency by time          1.25 s <

#  Memory estimate: 2.24 GiB, allocs estimate: 21.
# #+end_example


 ; @benchmark distM(X_1, X_2);



# #+RESULTS:
# #+begin_example
# BenchmarkTools.Trial: 2906 samples with 1 evaluation per sample.
#  Range (min … max):  1.444 ms …   4.001 ms  ┊ GC (min … max): 0.00% … 31.03%
#  Time  (median):     1.581 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   1.717 ms ± 337.199 μs  ┊ GC (mean ± σ):  6.26% ± 11.19%

#   ▂▇██▇▆▆▆▅▄▅▅▃▂             ▁▃▃▂▁    ▁▂▂▁                    ▂
#   ███████████████▇█▇▅▅▇▄▅▆▅▆▇███████▆███████▇▇▇▇▇▅▇▄▅▁▄▅▁▁▇▇▇ █
#   1.44 ms      Histogram: log(frequency) by time      2.95 ms <

#  Memory estimate: 3.86 MiB, allocs estimate: 1011.
# #+end_example


 ; using KernelAbstractions
@kernel function sparse_kernel(K)
Ind = @index(Global , Cartesian)
if abs(Ind[1] - Ind[2]) < 5
    K[Ind] = 1.
end
end;



# #+RESULTS:


 ; K = spzeros(10000,1000)
spkernel = sparse_kernel(CPU() , 256 , size(K))
spkernel(K);

# Kernel Implementation
# where \(\nabla_x , \Delta_x\) are the partial gradients and laplacians with respect to the second argument of \(k(x_j, \cdot )\).
# for a radial basis function \(\phi (r^2) \in  C^2(\RR)\)  and a corresponding RBF kernel \(k(x,x') := \phi (\frac{\|x-x'\|}{\gamma})\) they can be computed trivially
# \begin{align}
# \label{eq:2}
# \nabla_x k(x',x) &= \phi'\left(\frac{\|x - x'\|}{\gamma}\right) \cdot \frac{x - x'}{\gamma\|x - x'\|} \\
# \Delta_x k(x',x) &= \frac{1}{\gamma^2} \phi''\left(\frac{\|x - x'\|}{\gamma}\right) + \frac{1}{\gamma^{4}} \frac{d - 1}{\|x - x'\|} \cdot \phi'\left(\frac{\|x - x'\|}{\gamma}\right)
# \end{align}
# where \(d\) is the dimension of \(x\)

 ; using StaticArrays
function k(ϕ::Function , γ,x̂::SVector{N} ,x::SVector{N}) where N
    r = norm(x-x̂̂)/γ
    ϕ(r)
    end
function ∇k(dϕ::Function , γ ,x̂::SVector{N} ,x::SVector{N}) where N
    r = norm(x-x̂̂)/γ
    2*(x-x̂)/r*dϕ(r)
    end
function Δk(d²ϕ::Function, γ , dϕ::Function , x̂::SVector{N} ,x::SVector{N}) where N
    r = norm(x-x̂̂)/γ
    1/γ^2 * d²ϕ(r)  + 1/γ^2 * (N-1)/r*dϕ(r)
    end;



# #+RESULTS:
# : dd_rbf_gaussian (generic function with 3 methods)


 ; using GLMakie
X = range(-5 , 5 , 100)
Y = range(-5 , 5 , 100)
using LinearAlgebra

fig = Figure()
ax = Axis(fig[1,1])
lines!(X , x->rbf_gaussian(x))
save("images/gauss-rbf.png",fig );

# Cardinal B_{3} Spline

# \begin{align*}
# B_{d}(r) = \sum_{n=0}^4 \frac{(-1)^n}{d!} \binom{d+1}{n} \left( r + \frac{d+1}{2}-n \right)^d_+
# \end{align*}

 ; function B_3(r)
r_prime = r+2
    return 1/24 * (
    1 *max(0, (r_prime - 0)^3)
    -4*max(0, (r_prime - 1)^3)
    +6*max(0, (r_prime - 2)^3)
    -4*max(0, (r_prime - 3)^3)
    +1*max(0, (r_prime - 4)^3)
    )
end
function d_B_3(r)
r_prime = r+2
    return 1/24 * (
    1 *max(0, 3*(r_prime - 0)^2)
    -4*max(0, 3*(r_prime - 1)^2)
    +6*max(0, 3*(r_prime - 2)^2)
    -4*max(0, 3*(r_prime - 3)^2)
    +1*max(0, 3*(r_prime - 4)^2)
    )
end
function dd_B_3(r)
r_prime = r+2
    return 1/24 * (
    1 *max(0, 6*(r_prime - 0))
    -4*max(0, 6*(r_prime - 1))
    +6*max(0, 6*(r_prime - 2))
    -4*max(0, 6*(r_prime - 3))
    +1*max(0, 6*(r_prime - 4))
    )
end
;



# #+RESULTS:

# #+name: fig:b-spline

 ; using GLMakie
X = range(-2 , 2 , 100)
Y = range(-2 , 2 , 100)

fig = Figure()
ax = Axis(fig[1,1])

lines!(ax , X , B_3)

save("images/b-spline.png",fig );

# Thin Plate

 ; function thin_plate(r)
    r == 0.0 && return 0.0
    return r^2 * log(r)
end

function d_thin_plate(r)
    r == 0.0 && return 0.0
    return 2r * log(r) + r
end

function dd_thin_plate(r)
    r == 0.0 && return 0.0
    return 2 * log(r) + 3
end;



# #+RESULTS:
# : dd_thin_plate (generic function with 1 method)

# #+name: fig:plate-spline

 ; using GLMakie
X = range(0 , 1 , 100)
Y = range(-5 , 5 , 100)

fig = Figure()
ax = Axis(fig[1,1])

lines!(ax , X , thin_plate)

save("images/plate-spline.png",fig );

# PDE

 ; using Revise
includet("src/pdesolver.jl")
includet("src/domains.jl")
using .PDESolvers
using .Domains;

# Result


 ; using StaticArrays
function domain(x::SVector{2})
    return sdf_square(x , 0.5 , SVector(0.5,0.5))
end
function ∇domain(x::SVector{2})
    return sdf_square_grad(x , 0.5 , SVector(0.5,0.5))
end
function sdf_β(x::SVector{2})
    return sdf_square(x , 0. , SVector(-1.,-1) )
end

a(x::SVector{2}) = 1
∇a(x::SVector{2}) = SVector{2}(0.,0.)
f(x::SVector{2}) =2 * (x[1]+x[2] - x[1]^2 - x[2]^2)
g_D(x::SVector{2})= 0
g_N(x::SVector{2} , n::SVector{2}) = 0;

 ; γ = 100.
k_gauss(x,y) = k( rbf ,γ, x,y)
∇k_gauss(x,y) =∇k(d_rbf,γ , x,y)
Δk_gauss(x,y) = Δk(dd_rbf,γ , d_rbf, x,y)
S_gauss = PDESystem(k_gauss , ∇k_gauss , Δk_gauss , a, ∇a , f, g_D ,g_N , domain , ∇domain , sdf_β );

 ; k_plate(x,y) = k(thin_plate ,γ , x,y)
∇k_plate(x,y) =∇k(d_thin_plate ,γ , x,y)
Δk_plate(x,y) = Δk(dd_thin_plate,γ  , d_thin_plate , x,y)
S_plate = PDESystem(k_plate , ∇k_plate , Δk_plate , a, ∇a , f, g_D ,g_N , domain , ∇domain , sdf_β );

 ; k_bspline(x,y) = k(B_3,γ , x,y)
∇k_bspline(x,y) =∇k(d_B_3,γ , x,y)
Δk_bspline(x,y) = Δk(dd_B_3,γ , d_B_3 , x,y)
S_bspline = PDESystem(k_bspline , ∇k_bspline , Δk_bspline , a, ∇a , f, g_D ,g_N , domain , ∇domain , sdf_β );

 ; X = range(0 , 1 , 40)
Y = range(0 , 1 , 40)
X_col = [ [x,y] for x in X , y in Y]
X_col = reduce(vcat ,X_col )
X_col = reshape(X_col, 2,:)
X_t = range(0 , 1 , 100)
Y_t = range(0 , 1 , 100)
X_test = [ [x,y] for x in X_t , y in Y_t]
X_test = reduce(vcat , X_test)
X_test = reshape(X_test, 2,:)
size(X_col);



# #+RESULTS:
# : (2 1600)


 ; using LinearAlgebra
solution , K = solve(S_gauss ,X_col)
cond(K);



# #+RESULTS:
# : julia-async:04fba19e-2cf6-489d-b224-77764e3aaa24


# #+name: fig:solution

 ; using GLMakie
fig = Figure()
ax = Axis(fig[1,1] , title="Aproximate solution")
sol , K = solution(X_test)
sol = reshape(sol , size(X_t,1) , :)
hm = heatmap!(ax , X,Y, sol)
Colorbar(fig[:, end+1], hm)
save("images/solution.png",fig );



# #+RESULTS: fig:solution
# [[file:images/solution.png]]

# #+name: fig:exact-solution

 ; using GLMakie
u(x , y) = x * (1-x) * y* ( 1- y)
u(x) = u(x[1] , x[2])
fig = Figure()
ax = Axis(fig[1,1])

hm = heatmap!(ax,X_t,Y_t,u)
Colorbar(fig[:, end+1], hm)
save("images/exact-solution.png",fig );



# #+RESULTS: fig:exact-solution
# [[file:images/exact-solution.png]]


 ; sol , _ = solution(X_test)
norm(sol - u.(eachcol(X_test)) , Inf);

# Result
# where

 ; using StaticArrays
a(x::SVector{2}) = x[1] + 2
∇a(x::SVector{2}) = SVector{2}(1.,0.)
α = 2.
β = 1.5
f(x::SVector{2} , ::Val{α}) where α = - α*norm(x ,2)^(α - 2)*(3x[1] +4) - α*(α -2) * (x[1] + 2) * norm(x,2)^(α - 3)
g_D(x::SVector{2} , ::Val{α}) where α = norm(x,2)^α
g_N(x::SVector{2} , n::SVector{2} , ::Val{α}) where α = α* norm(x,2.)^(α-2.)*(x[1] +2.) * x ⋅ n
f(x) = f(x,Val(α))
g_D(x) = g_D(x,Val(α))
g_N(x, n) = g_N(x , n,Val(α))
function sdf_β(x::SVector{2})
    return sdf_square(x , β , SVector(-1.,-1) )
end
S = PDESystem(k_gauss , ∇k_gauss , Δk_gauss , a, ∇a , f, g_D ,g_N , sdf_L , sdf_L_grad , sdf_β );



# #+RESULTS:
# : PDESystem(Main.k_gauss, Main.∇k_gauss, Main.Δk_gauss, Main.a, Main.∇a, Main.f, Main.g_D, Main.g_N, Main.Domains.sdf_L, Main.Domains.sdf_L_grad, Main.Domains.sdf_β)


 ; X = range(-1 , 1 , 30)
Y = range(-1 , 1 , 30)
X_col = [ [x,y] for x in X , y in Y]
X_col = reduce(vcat ,X_col )
X_col = reshape(X_col, 2,:)
X_t = range(-2 , 2 , 100)
Y_t = range(-2, 2 , 100)
X_test = [ [x,y] for x in X_t , y in Y_t]
X_test = reduce(vcat , X_test)
X_test = reshape(X_test, 2,:)
size(X_col);



# #+RESULTS:
# : (2 900)


 ; using LinearAlgebra
solution , K = solve(S ,X_col)
cond(K);



# #+RESULTS:
# : 221981.19f0

# #+name: fig:diffusion-solution

 ; using GLMakie
fig = Figure()
ax = Axis(fig[1,1] , title="Aproximate solution")
sol , K = solution(X_test)
sol = reshape(sol , size(X_t,1) , :)
hm = heatmap!(ax , X,Y, sol)
Colorbar(fig[:, end+1], hm)
save("images/diffusion-solution.png",fig );

 ; using Random
using CUDA
dev = CUDA.functional() ? cu : Array
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
