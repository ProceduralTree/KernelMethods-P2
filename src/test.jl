using KernelAbstractions: ndrange

using KernelAbstractions
using StaticArrays
using CUDA

@kernel function assemble!(A,@Const(X))
    I = @index(Global, Cartesian)
    A[I] = X[I[1],1] * X[I[2],1] + X[I[1],2] * X[I[2],2] +X[I[1],3] * X[I[2],3]
    end
@kernel function assemble2!(A,@Const(X))
    I = @index(Global, Cartesian)
    # devilish array access inside GPU Kernel.
    x = @inbounds SVector{3}(view(X , I[1] , :))
    x̂ = @inbounds SVector{3}(view(X , I[2] , :))
    A[I] = x' * x̂

    end
dev = cu
n = 10
A = zeros(Float32,10,10) |> cu
X = rand(Float32 , n , 3) |> cu
d = get_backend(X)

assemble = assemble!(d, 64,(n,n))
assemble2 = assemble2!(d, 64,(n,n))
assemble(A,X , ndrange=size(A))
assemble2(A,X , ndrange=size(A).-1)
A
