module TransposeUtils

export CPU, GPU
export on_device, device, set!
export Partition, TransposableArrays, DistributedTransform
export transpose_x_to_y!, transpose_y_to_x!, transpose_y_to_z!, transpose_z_to_y!
export @distributed_info

using CUDA
using MPI
using FFTW
using KernelAbstractions
using KernelAbstractions: @index, @kernel
using MPI: VBuffer, Alltoallv!

import Base

const CPU = KernelAbstractions.CPU
const GPU = CUDA.CUDABackend

@inline device(::CuArray) = CUDA.CUBackend()
@inline device(::Array)   = KernelAbstractions.CPU()

@inline sync_device!(::GPU) = CUDA.synchronize()
@inline sync_device!(::CPU) = nothing

@inline on_device(::GPU, a::CuArray) = a
@inline on_device(::GPU, a::Array)   = CUDA.CuArray(a)
@inline on_device(::CPU, a::CuArray) = Array(a)
@inline on_device(::CPU, a::Array)   = a

@inline Base.zeros(::CPU, T, dims...) = zeros(T, dims...)
@inline Base.zeros(::GPU, T, dims...) = CUDA.zeros(T, dims...)

include("partition.jl")

@inline on_device(p::Partition, a) = on_device(device(p), a)
@inline Base.zeros(p::Partition, T, dims...) = zeros(device(p), T, dims...)  

set!(array, func) = _set!(device(array), (16, 16), size(array))(array, func)

@kernel function _set!(array, func::Function)
    i, j, k = @index(Global, NTuple)
    @inline array[i, j, k] = func(i, j, k)
end

@kernel function _set!(array1, array2::AbstractArray)
    i, j, k = @index(Global, NTuple)
    @inline array1[i, j, k] = array2[i, j, k]
end

@inline global_size(p::Partition, dims) = map(sum, concatenate_local_sizes(dims, p))

macro distributed_info(expr)
    handshake = quote
        for rank in 0:MPI.Comm_size(MPI.COMM_WORLD) - 1
            if MPI.Comm_rank(MPI.COMM_WORLD) == rank
                @info "rank $rank" $expr
            end
            MPI.Barrier(MPI.COMM_WORLD)
        end
    end
    return :($(esc(handshake)))
end

include("transposable_arrays.jl")
include("distributed_transpose.jl")
include("distributed_fourier_transform.jl")

end