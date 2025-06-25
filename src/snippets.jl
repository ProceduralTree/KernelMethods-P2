# Distance Matrix Computation :noexport:

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



# #+RESULTS:
# : julia-async:f3f21d6e-71df-4891-8781-f0dd47c6dd10


 ; using GLMakie
X = range(-2 , 2 , 100)
using LinearAlgebra

fig = Figure()
ax = Axis(fig[1,1])
lines!(ax , X ,x->    rbf_gaussian(x^2), label=L"gauss")
lines!(ax , X ,x-> -2x* d_rbf_gaussian(x^2) , label=L"\partial gauss")
lines!(ax , X ,x-> 4x^2* dd_rbf_gaussian(x^2) , label=L"\partial^2 gauss")
axislegend(ax)
save("images/gauss-rbf.png",fig );



# #+name: fig:b-spline

 ; using GLMakie
using LaTeXStrings
X = range(-2 , 2 , 100)
Y = range(-2 , 2 , 100)

fig = Figure()
ax = Axis(fig[1,1])

lines!(ax , X , B_3 , label=L"B_3")
lines!(ax , X , d_B_3 , label=L"\partial B_3")
lines!(ax , X , dd_B_3 , label=L"\partial^2 B_3")
axislegend(ax)
save("images/b-spline.png",fig );



# #+RESULTS:
# : dd_thin_plate (generic function with 1 method)

# #+name: fig:plate-spline

 ; using GLMakie
X = range(0 , 1 , 100)
Y = range(-5 , 5 , 100)

fig = Figure()
ax = Axis(fig[1,1])

lines!(ax , X , x-> x^2 * log(x),  label=L"T")
lines!(ax , X , x-> 2x * log(x) + x , label=L"\partial T")
lines!(ax , X , x-> 2log(x) + 3 , label=L"\partial^2 T")
axislegend(ax)

save("images/plate-spline.png",fig );

# PDE
# To use our PDE solver we include all our modules

 ; using Revise
using LinearAlgebra
includet("src/pdesolver.jl")
includet("src/domains.jl")
includet("src/rbf.jl")
using .PDESolvers
using .Domains
using .RadialBasisKernels;



# #+RESULTS:

# and generate a set of collocation and test points. If a functional CUDA GPU is available, we move the data to the GPU. The solver will then attempt so solve on the GPU. Anoyingly all functions have to be known at compile time, when using the GPU backend.

 ; using CUDA
dev = CUDA.functional() ? cu : Array
#dev = Array
X = range(0 , 1 , 20)
Y = range(0 , 1 , 20)
X_col = [ [x,y] for x in X , y in Y]
X_col = reduce(vcat ,X_col )
X_col = reshape(X_col, 2,:) |> dev
X_t = range(0 , 1 , 100)
Y_t = range(0 , 1 , 100)
X_test = [ [x,y] for x in X_t , y in Y_t]
X_test = reduce(vcat , X_test)
X_test = reshape(X_test, 2,:) |> dev
X_lol = rand(2,400) |> dev


size(X_col);

# PDE Poisson
# with \(a(x) = 1 , g_{D}(x) = 0\) and \(\Gamma_{N} = \emptyset \) this method is able to model the poisson equation
# \begin{align}
# \label{eq:poisson}
# - \Delta u(x) &= f(x) & \text{in} \quad \Omega \\
# u(x) &= 0 & \text{on} \quad  \Gamma_D
# \end{align}

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

# Plotting Utility

 ; using LaTeXStrings
function plot(fig , i,::Val{γ} , limits , errors, rbf , d_rbf , dd_rbf) where γ
        @inline k_rbf(x,y) = @inline k( rbf ,Val(γ), x,y)
        @inline ∇k_rbf(x,y) =@inline ∇k(d_rbf,Val(γ), x,y)
        @inline Δk_rbf(x,y) =@inline Δk(dd_rbf , d_rbf ,Val(γ), x,y)
        S_gauss = PDESystem(k_rbf , ∇k_rbf , Δk_rbf , a, ∇a , f, g_D ,g_N , domain , ∇domain , sdf_β )
        solution , K = solve(S_gauss ,X_col);
        ax = Axis(fig[1,i] , title=L"$\gamma=%$γ$ Condition %$(cond(K))", aspect=DataAspect())
        sol , K_t = solution(X_test)
        push!(errors , norm(Array(sol) - u.(eachcol(Array(X_test))) , Inf))
        sol = reshape(Array(sol) , size(X_t,1) , :)
        hm = heatmap!(ax , X,Y, sol , colorrange=limits)
        return fig
end
function plotsq(fig , i,::Val{γ} , limits , errors, rbf , d_rbf , dd_rbf) where γ
        @inline k_rbf(x,y) = @inline ksq( rbf ,Val(γ), x,y)
        @inline ∇k_rbf(x,y) =@inline ∇ksq(d_rbf,Val(γ), x,y)
        @inline Δk_rbf(x,y) =@inline Δksq(dd_rbf , d_rbf ,Val(γ), x,y)
        S_gauss = PDESystem(k_rbf , ∇k_rbf , Δk_rbf , a, ∇a , f, g_D ,g_N , domain , ∇domain , sdf_β )
        solution , K = solve(S_gauss ,X_col);
        ax = Axis(fig[1,i] , title=L"$\gamma=%$γ$ Condition %$(cond(K))", aspect=DataAspect())
        sol , K_t = solution(X_test)
        push!(errors , norm(Array(sol) - u.(eachcol(Array(X_test))) , Inf))
        sol = reshape(Array(sol) , size(X_t,1) , :)
        hm = heatmap!(ax , X,Y, sol , colorrange=limits)
        return fig
end;

# Results
# #+name: fig:gauss-kernel

 ; using GLMakie
fig = Figure(size=(2600,400))
limits = (0, 0.06)
errors = Vector{Float32}()
u(x , y) = x * (1-x) * y* ( 1- y)
u(x) = u(x[1] , x[2])
for (i,gamma) in enumerate([Val(0.1) , Val(0.075) , Val(0.05) , Val(0.025)])
plotsq(fig , i,gamma , limits , errors , rbf_gaussian , d_rbf_gaussian , dd_rbf_gaussian)
end
ax = Axis(fig[1,0] , title="Exact sollution" , aspect=DataAspect())
hm = heatmap!(ax,X_t,Y_t,u , colorrange=limits)
Colorbar(fig[:, end+1], hm)
ax = Axis(fig[1,end+1] , title=L"$L^\infty$ Error" , xlabel=L"\gamma" , ylabel=L"|u_h - u |_\infty")
lines!(ax , [0.01 , 0.0075 , 0.005 , 0.0025] , errors)
save("images/gauss-kernel.png",fig );



# #+caption: gauss kernel with various values for \(\gamma \)
# #+RESULTS: fig:gauss-kernel
# [[file:images/gauss-kernel.png]]

# #+name: fig:thin-plate-kernel

 ; using GLMakie
fig = Figure(size=(2600,400))
limits = (0, 0.06)
errors = Vector{Float32}()
for (i,gamma) in enumerate([Val(1.0) , Val(0.1) , Val(0.05) , Val(0.01)])
plotsq(fig , i,gamma , limits , errors , thin_plate , d_thin_plate , dd_thin_plate)
end
u(x , y) = x * (1-x) * y* ( 1- y)
u(x) = u(x[1] , x[2])
ax = Axis(fig[1,0] , title="Exact sollution" , aspect=DataAspect())
hm = heatmap!(ax,X_t,Y_t,u , colorrange=limits)
Colorbar(fig[:, end+1], hm)
ax = Axis(fig[1,end+1] , title=L"$L^\infty$ Error" , xlabel=L"\gamma" , ylabel=L"|u_h - u |_\infty")
lines!(ax , [1.,0.1,0.05,0.01] , errors)
save("images/thin-plate-kernel.png",fig );



# #+RESULTS: fig:thin-plate-kernel
# [[file:images/thin-plate-kernel.png]]
# #+name: fig:B3-spline-kernel

 ; using GLMakie
fig = Figure(size=(2600,400))
limits = (0, 0.06)
errors = Vector{Float32}()
for (i,gamma) in enumerate([Val(0.5) , Val(1.0) , Val(1.5) , Val(2.0)])
plot(fig , i,gamma , limits , errors , B_3 , d_B_3 , dd_B_3)
end
u(x , y) = x * (1-x) * y* ( 1- y)
u(x) = u(x[1] , x[2])
ax = Axis(fig[1,0] , title="Exact sollution" , aspect=DataAspect())
hm = heatmap!(ax,X_t,Y_t,u , colorrange=limits)
Colorbar(fig[:, end+1], hm)
ax = Axis(fig[1,end+1] , title=L"$L^\infty$ Error" , xlabel=L"\gamma" , ylabel=L"|u_h - u |_\infty")
lines!(ax , [0.5,1.,1.5,2.] , errors)
save("images/B3-spline-kernel.png",fig );

# Diffusion PDE
# we evaluate the diffusion PDE with
# \begin{align*}
# a(x) &=  x_1 +2 \\
# f(x) &= - \alpha \|x\|^{\alpha -2} * (3x_1 + 4) - \alpha * (\alpha -2) * (x_1 +2) * \|x\|^{\alpha -3} \\
# g_D(x) &= \|x\|^{\alpha }\\
# g_{N}(x,n) &= \alpha  \|x\|^{\alpha -2}* (x_1 +2) * x \cdot  n
# \end{align*}
# where

 ; using StaticArrays
a(x::SVector{2}) = x[1] + 2
∇a(x::SVector{2}) = SVector{2}(1.,0.)
α = 0.5
β = 0.2
f(x::SVector{2} , ::Val{α}) where α = - α*norm(x ,2)^(α - 2)*(3x[1] +4) - α*(α -2) * (x[1] + 2) * norm(x,2)^(α - 3)
g_D(x::SVector{2} , ::Val{α}) where α = norm(x,2)^α
g_N(x::SVector{2} , n::SVector{2} , ::Val{α}) where α = α* norm(x,2.)^(α-2.)*(x[1] +2.) * x ⋅ n
f(x) = f(x,Val(α))
g_D(x) = g_D(x,Val(α))
g_N(x, n) = g_N(x , n,Val(α))
function sdf_β(x::SVector{2})
    return sdf_square(x , β , SVector(-1.,-1) )
end;



# #+RESULTS:
# : sdf_β (generic function with 1 method)

# And select a collocation set filtered to be inside the domain

 ; X = range(-1 , 1 , 11)
Y = range(-1 , 1 , 11)
X_col = [ [x,y] for x in X , y in Y]
X_col = reduce(vcat ,X_col )
X_col = reshape(X_col, 2,:)
X_col = filter(x -> sdf_L(SVector{2}(x)) <= 0 , eachcol(X_col))
X_col = reduce(hcat , X_col) |> dev
X_t = range(-1.1 , 1.1 , 100)
Y_t = range(-1.1, 1.1 , 100)
X_test = [ [x,y] for x in X_t , y in Y_t]
X_test = reduce(vcat , X_test)
X_test = reshape(X_test, 2,:) |> dev
size(X_col);

# Plotting utility

 ; using LaTeXStrings
function plotdiff(fig , i,::Val{γ} , limits ,  rbf , d_rbf , dd_rbf) where γ
        @inline k_rbf(x,y) = @inline k( rbf ,Val(γ), x,y)
        @inline ∇k_rbf(x,y) =@inline ∇k(d_rbf,Val(γ), x,y)
        @inline Δk_rbf(x,y) =@inline Δk(dd_rbf , d_rbf ,Val(γ), x,y)
        S_gauss = PDESystem(k_rbf , ∇k_rbf , Δk_rbf , a, ∇a , f, g_D ,g_N , sdf_L , sdf_L_grad , sdf_β )
        solution , K = solve(S_gauss ,X_col);
        ax = Axis(fig[1,i] , title=L"$\gamma=%$γ$ Condition %$(cond(K))", aspect=DataAspect())
        sol , K_t = solution(X_test)
        push!(errors , norm(Array(sol) - u.(eachcol(Array(X_test))) , Inf))
        sol = reshape(Array(sol) , size(X_t,1) , :)
        hm = heatmap!(ax , X,Y, sol , colorrange=limits)
        return fig
end
function plotsqdiff(fig , i,::Val{γ} , limits , rbf , d_rbf , dd_rbf) where γ
        @inline k_rbf(x,y) = @inline ksq( rbf ,Val(γ), x,y)
        @inline ∇k_rbf(x,y) =@inline ∇ksq(d_rbf,Val(γ), x,y)
        @inline Δk_rbf(x,y) =@inline Δksq(dd_rbf , d_rbf ,Val(γ), x,y)
        S_gauss = PDESystem(k_rbf , ∇k_rbf , Δk_rbf , a, ∇a , f, g_D ,g_N , sdf_L , sdf_L_grad , sdf_β )
        solution , K = solve(S_gauss ,X_col);
        ax = Axis(fig[1,i] , title=L"$\gamma=%$γ$ Condition %$(cond(K))", aspect=DataAspect())
        sol , K_t = solution(X_test)
        push!(errors , norm(Array(sol) - u.(eachcol(Array(X_test))) , Inf))
        sol = reshape(Array(sol) , size(X_t,1) , :)
        hm = heatmap!(ax , X,Y, sol , colorrange=limits)
        return fig
end;

# Results
# #+name: fig:gauss-kernel-diff

 ; using GLMakie
fig = Figure(size=(1800,400))
limits = (-0.6, 0)
for (i,gamma) in enumerate([Val(0.1) , Val(0.075) , Val(0.05) , Val(0.025)])
plotsqdiff(fig , i,gamma , limits , rbf_gaussian , d_rbf_gaussian , dd_rbf_gaussian)
end
Colorbar(fig[:, end+1], hm)
save("images/gauss-kernel-diff.png",fig );



# #+caption: gauss kernel with various values for \(\gamma \) for a diffusive system
# #+RESULTS: fig:gauss-kernel-diff
# [[file:images/gauss-kernel-diff.png]]



# #+name: fig:thin-plate-kernel-diff

 ; fig = Figure(size=(1800,400))
limits = (-0.6, 0)
for (i,gamma) in enumerate([Val(0.1) , Val(0.075) , Val(0.05) , Val(0.025)])
plotsqdiff(fig , i,gamma , limits , thin_plate , d_thin_plate , dd_thin_plate)
end
Colorbar(fig[:, end+1], hm)
save("images/thin-plate-kernel-diff.png",fig );



# #+caption: thin plate kernel with various values for \(\gamma \) for a diffusive system
# #+RESULTS: fig:thin-plate-kernel-diff
# [[file:images/thin-plate-kernel-diff.png]]

# #+name: fig:B3-spline-kernel-diff

 ; using GLMakie
fig = Figure(size=(1800,400))
limits = (-0.6, 0)
for (i,gamma) in enumerate([Val(1.0) , Val(1.25) , Val(1.5) , Val(1.75)])
plotsqdiff(fig , i,gamma , limits , B_3 , d_B_3 , dd_B_3)
end
Colorbar(fig[:, end+1], hm)
save("images/B3-spline-kernel-diff.png",fig );
