

# #+RESULTS:


 ; test();



# #+RESULTS:
# : hello world


 ; 
using Plots
f(x) = exp(x) + 10 * sin(x)
plot(f)
savefig("images/plot.png");



# #+RESULTS:
# [[file:images/plot.png]]

 ; using KernelAbstractions

@kernel function Δ_kernel(A , Δ)
    I = @index(Global , Cartesian)
    Ix = oneunit(I)
    Δ[I] = 0
    dimensions = ndims(A)
    for i in 1:dimensions
        zero(I)
        In = CartesianIndex()
        Δ[I] += A[I + In] - A[I]
    end
end;



# #+RESULTS:
# : Δ_kernel (generic function with 4 methods)



 ; dt = 0.1
for t in 0:0.1:10
x += dx * dt
end;

# Kernel Implementation Implementation

 ; function rbf_gaussian(r, ::Val{γ}) where γ
    exp(- γ * r)
    end
function rbf_gaussian′(r , ::Val{γ}) where γ
    - γ * exp(- γ * r)
    end
function rbf_gaussian′′(r , ::Val{γ}) where γ
    γ^2 * exp(- γ * r)
    end
function k_gauss(x , x̂)
    rbf_gaussian(norm(x-x̂) , Val(0.1))
    end
function ∇k_gauss(x,x̂)
    rbf_gaussian′(norm(x-x̂) , Val(0.1)) * x
    end
function Δk_gauss(x,x̂)
    rbf_gaussian′′(norm(x-x̂) , Val(0.1))  + rbf_gaussian′(norm(x-x̂) , Val(0.1)) *  dot(x , x)
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
surface!(ax, x,y, z)
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

 ; a(x::SVector{3}) = x[1] + 2
∇a(x::SVector{3}) = SVector{3}(1.,0.,0.);



# #+RESULTS:
# : assemble_matrix! (generic function with 4 methods)


 ; X = rand(3,10)
A = rand(10,10)
assemble_matrix!(A  , ;
