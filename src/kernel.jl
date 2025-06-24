# Preamble :noexport:

 ; 
module Kernel
using StaticArrays
using KernelAbstractions
using LinearAlgebra;

# Regression Approach
# Aim of this excrcise is to find solutions \(u\in \mathcal{H}_k\) such that they satisfy the following system

# \begin{align}
# \label{eq:pde}
# - \nabla \cdot   \left( a(x) \nabla u(x) \right) &= f(x) & \text{in} \quad \Omega \\
# u(x) &= g_D(x) & \text{on} \quad  \Gamma_D \\
# \left( a(x) \nabla u(x)  \right) \cdot  \vec{n}(x) &= g_N & \text{on} \quad \Gamma_N
# \end{align}
# we do this by projecting the system onto \(\mathcal{H}_k(\Omega)\)
# \begin{align}
# \label{eq:pde_proj}
# \left<   - \nabla \cdot   \left( a(x) \nabla u(x) \right),\phi \right>&= \left< f(x) ,\phi  \right> & \text{in} \quad \Omega , \phi \in  \mathcal{H}_{k} \\
# \left<   u(x) , \phi \right>&= \left< g_D(x) , \phi  \right> & \text{on} \quad  \Gamma_D \\
# \left<   \left( a(x) \nabla u(x)  \right) \cdot  \vec{n}(x) , \phi \right>&= \left< g_N ,\phi  \right> & \text{on} \quad \Gamma_N
# \end{align}
# Let \( \hat{X}:=\left\{ x_j \right\}_{j=1}^n \subset\RR ^d\). Since \(\left\{ k(x_i,\cdot ) \right\}_{i=1}^n\) is a basis of \(\mathcal{H}_k\) it also has to hold
# \begin{align}
# \label{eq:pde_proj}
# \left<   - \nabla \cdot   \left( a(x) \nabla u(x) \right),k(x_i,\cdot ) \right>&= \left< f(x) ,k(x_i,\cdot )  \right> & \text{in} \quad \Omega , x_i \in  X \\
# \left<   u(x) , k(x_i,\cdot ) \right>&= \left< g_D(x) , k(x_i,\cdot )  \right> & \text{on} \quad  \Gamma_D \\
# \left<   \left( a(x) \nabla u(x)  \right) \cdot  \vec{n}(x) , k(x_i,\cdot ) \right>&= \left< g_N , k(x_i , \cdot )  \right> & \text{on} \quad \Gamma_N
# \end{align}
# We assuming \(f,g_D , g_N(\cdot ,\vec{n}) \in  \mathcal{H}_k\) i.e. \(\left< f , k(x_i , \cdot ) \right> = f(x_i)\) etc. We search for a finite approximation \(u_h \approx u\)
#  such that it satisfies \eqref{eq:pde_proj} where
# \begin{align}
# \label{eq:approx}
# u_h(x) &= \sum_{j=1}^{n} a_j k(x_j,x)
# \end{align}
# correspondingly we are able to directly compute

# \begin{align*}
# \nabla_x u_h(x) &= \sum_{j=1}^n a_j \nabla_x  k(x_j ,x) \\
# - \nabla_x \cdot \left( a(x) \nabla_x u_h(x) \right) &= -  \nabla_x a(x) \cdot  \nabla_x u(x)  - a(x) \Delta_x u(x) \\
# &=  - \sum_{j=1}^{n} a_j \left(  \nabla_x a(x) \cdot  \nabla_x k(x_j,x)   + a(x) \Delta_x k(x_j,x)\right)
# \end{align*}
# this leads to the following Linear system
# \begin{align}
# \label{eq:pde-sys}
#  - \sum_{j=1}^{n} a_j \left(  \nabla_{x_i} a(x_i) \cdot  \nabla_{x_i} k(x_j,x_i)   + a(x_i) \Delta_{x_i} k(x_j,x_i)\right)&=  f(x_i)  & x_i\in  \Omega , x_i \in  X \\
#  \sum_{j=1}^{n} a_j k(x_j,x_i)&=  g_D(x_i) & x_i\in   \Gamma_D \\
# \sum_{j=1}^n  a_j \left( a(x_i) \nabla_{x_i}  k(x_j ,x_i) \cdot  n_i \right) &=  g_N(x_i , n_i) & x_i \in  \Gamma_N
# \end{align}

# this corresponds directly with the System Matrix \(K\), that we compute in julia using a GPU copatible kernel that employs element wise notation

 ; 
@kernel function system_matrix!(K ,@Const(X), a , ∇a ,k, ∇k, Δk  , sdf , grad_sdf , sdf_beta)
    Iᵢⱼ =  @index(Global , Cartesian)
    @inbounds xᵢ= SVector{2}(view(X, : , Iᵢⱼ[1])) # Essentially X[:,i]
    @inbounds xⱼ= SVector{2}(view(X, : , Iᵢⱼ[2])) # Essentially X[:,j]
    # poisson equation
    @inbounds K[Iᵢⱼ] = -a(xᵢ)*Δk(xᵢ,xⱼ)- ∇a(xᵢ)⋅∇k(xᵢ,xⱼ)
    if abs(sdf(xᵢ)) < 1e-10
        if sdf_beta(xᵢ) < 0
            # Neumann Boundary Condition
            @inbounds nᵢ= grad_sdf(xᵢ)
            @inbounds K[Iᵢⱼ] = a(xᵢ) * (nᵢ ⋅ ∇k(xᵢ , xⱼ))
        else
          # Dirichlet Boundary
          @inbounds K[Iᵢⱼ] =k(xᵢ , xⱼ)
        end
    end
end;

# right hand side
# The right hand side of the system is computed in a similar Fashion

 ; 
@kernel function apply_function_colwise!(B ,@Const(X) , f , g_D , g_N , sdf  , grad_sdf, sdf_beta)
    # boilerplate
    Iᵢ = @index(Global , Cartesian)
    @inbounds xᵢ= SVector{2}(view(X , : , Iᵢ[1]))
    # poisson equation

    @inbounds B[Iᵢ] = f(xᵢ)
    if abs(sdf(xᵢ)) < 1e-10
         if sdf_beta(xᵢ) < 0
             # Neumann Boundary Condition
             @inbounds nᵢ= grad_sdf(xᵢ)
             @inbounds B[Iᵢ] = g_N(xᵢ , nᵢ )
         else
            # Dirichlet Boundary
            @inbounds B[Iᵢ] = g_D(xᵢ)
         end
     end
end;

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

# Postamble

 ; 
export linear_matrix!
export dirichlet_matrix!
export neumann_matrix!
export apply_function_colwise!
export system_matrix!
end;
