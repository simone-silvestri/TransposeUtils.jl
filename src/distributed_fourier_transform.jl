using FFTW
import FFTW: fft, ifft

struct DistributedTransform{T, P}
    arrays :: T
    plans :: P
end

function plan_forward_transform(A::Array, dims, planner_flag=FFTW.PATIENT)
    length(dims) == 0 && return nothing
    return FFTW.plan_fft!(A, dims, flags=planner_flag)
end

function plan_forward_transform(A::CuArray, dims, planner_flag=FFTW.PATIENT)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_fft!(A, dims)
end

function plan_backward_transform(A::Array, dims, planner_flag=FFTW.PATIENT)
    length(dims) == 0 && return nothing
    return FFTW.plan_ifft!(A, dims, flags=planner_flag)
end

function plan_backward_transform(A::CuArray, dims, planner_flag=FFTW.PATIENT)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_ifft!(A, dims)
end

function DistributedTransform(arrays::TransposableArrays)
    plan_x = (forward  =  plan_forward_transform(arrays.xfield, [1]),
              backward = plan_backward_transform(arrays.xfield, [1]))
    plan_y = (forward  =  plan_forward_transform(arrays.yfield, [2]),
              backward = plan_backward_transform(arrays.yfield, [2]))
    plan_z = (forward  =  plan_forward_transform(arrays.zfield, [3]),
              backward = plan_backward_transform(arrays.zfield, [3]))

    plans = (x = plan_x, y = plan_y, z = plan_z)

    DistributedTransform(arrays, plans)
end

@inline function fft(transform::DistributedTransform)
    plans  = transform.plans
    arrays = transform.arrays
    
    plans.z.forward * arrays.zfield
    transpose_z_to_y!(arrays)
    plans.y.forward * arrays.yfield
    transpose_y_to_x!(arrays)
    plans.x.forward * arrays.xfield

    return nothing
end

@inline function ifft(transform::DistributedTransform)
    plans  = transform.plans
    arrays = transform.arrays
    
    plans.x.backward * arrays.xfield
    transpose_x_to_y!(arrays)
    plans.y.backward * arrays.yfield
    transpose_y_to_z!(arrays)
    plans.z.backward * arrays.zfield
    
    return nothing
end