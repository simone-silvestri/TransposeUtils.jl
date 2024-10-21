
# Fallbacks for slab decompositions
transpose_z_to_y!(::SlabYArrays) = nothing
transpose_y_to_z!(::SlabYArrays) = nothing
transpose_x_to_y!(::SlabXArrays) = nothing
transpose_y_to_x!(::SlabXArrays) = nothing

#####
##### Packing and unpacking buffers for MPI communication
#####

@kernel function _pack_buffer_z_to_y!(yzbuff, zfield, N)
    i, j, k = @index(Global, NTuple)
    Nx, Ny, _ = N
    @inbounds yzbuff.send[j + Ny * (i-1 + Nx * (k-1))] = zfield[i, j, k]
end

@kernel function _pack_buffer_x_to_y!(xybuff, xfield, N)
    i, j, k = @index(Global, NTuple)
    _, Ny, Nz = N
    @inbounds xybuff.send[j + Ny * (k-1 + Nz * (i-1))] = xfield[i, j, k]
end

# packing a y buffer for communication with a x-local direction (y -> x communication)
@kernel function _pack_buffer_y_to_x!(xybuff, yfield, N) 
    i, j, k = @index(Global, NTuple)
    Nx, _, Nz = N
    @inbounds xybuff.send[i + Nx * (k-1 + Nz * (j-1))] = yfield[i, j, k]
end

# packing a y buffer for communication with a z-local direction (y -> z communication)
@kernel function _pack_buffer_y_to_z!(xybuff, yfield, N) 
    i, j, k = @index(Global, NTuple)
    Nx, _, Nz = N
    @inbounds xybuff.send[k + Nz * (i-1 + Nx * (j-1))] = yfield[i, j, k]
end

@kernel function _unpack_buffer_x_from_y!(xybuff, xfield, N, n)
    i, j, k = @index(Global, NTuple)
    size = n[1], N[2], N[3]
    @inbounds begin
        i′  = mod(i - 1, size[1]) + 1
        m   = (i - 1) ÷ size[1]
        idx = i′ + size[1] * (k-1 + size[3] * (j-1)) + m * prod(size)
        xfield[i, j, k] = xybuff.recv[idx]
    end
end

@kernel function _unpack_buffer_z_from_y!(yzbuff, zfield, N, n)
    i, j, k = @index(Global, NTuple)
    size = N[1], N[2], n[3]
    @inbounds begin
        k′  = mod(k - 1, size[3]) + 1
        m   = (k - 1) ÷ size[3]
        idx = k′ + size[3] * (i-1 + size[1] * (j-1)) + m * prod(size)
        zfield[i, j, k] = yzbuff.recv[idx]
    end
end

# unpacking a y buffer from a communication with z-local direction (z -> y)
@kernel function _unpack_buffer_y_from_z!(yzbuff, yfield, N, n) 
    i, j, k = @index(Global, NTuple)
    size = N[1], n[2], N[3]
    @inbounds begin
        j′  = mod(j - 1, size[2]) + 1
        m   = (j - 1) ÷ size[2]
        idx = j′ + size[2] * (i-1 + size[1] * (k-1)) + m * prod(size)
        yfield[i, j, k] = yzbuff.recv[idx]
    end
end

# unpacking a y buffer from a communication with x-local direction (x -> y)
@kernel function _unpack_buffer_y_from_x!(yzbuff, yfield, N, n) 
    i, j, k = @index(Global, NTuple)
    size = N[1], n[2], N[3]
    @inbounds begin
        j′  = mod(j - 1, size[2]) + 1
        m   = (j - 1) ÷ size[2] 
        idx = j′ + size[2] * (k-1 + size[3] * (i-1)) + m * prod(size)
        yfield[i, j, k] = yzbuff.recv[idx]
    end
end

pack_buffer_x_to_y!(buff, f) = _pack_buffer_x_to_y!(device(f), (16, 16), size(f))(buff, f, size(f))
pack_buffer_z_to_y!(buff, f) = _pack_buffer_z_to_y!(device(f), (16, 16), size(f))(buff, f, size(f))
pack_buffer_y_to_x!(buff, f) = _pack_buffer_y_to_x!(device(f), (16, 16), size(f))(buff, f, size(f))
pack_buffer_y_to_z!(buff, f) = _pack_buffer_y_to_z!(device(f), (16, 16), size(f))(buff, f, size(f))

unpack_buffer_x_from_y!(f, fo, buff) = _unpack_buffer_x_from_y!(device(f), (16, 16), size(f))(buff, f, size(f), size(fo))
unpack_buffer_z_from_y!(f, fo, buff) = _unpack_buffer_z_from_y!(device(f), (16, 16), size(f))(buff, f, size(f), size(fo))
unpack_buffer_y_from_x!(f, fo, buff) = _unpack_buffer_y_from_x!(device(f), (16, 16), size(f))(buff, f, size(f), size(fo))
unpack_buffer_y_from_z!(f, fo, buff) = _unpack_buffer_y_from_z!(device(f), (16, 16), size(f))(buff, f, size(f), size(fo))

for (from, to, buff) in zip([:y, :z, :y, :x], [:z, :y, :x, :y], [:yz, :yz, :xy, :xy])
    transpose!      = Symbol(:transpose_, from, :_to_, to, :(!))
    pack_buffer!    = Symbol(:pack_buffer_, from, :_to_, to, :(!)) 
    unpack_buffer!  = Symbol(:unpack_buffer_, to, :_from_, from, :(!)) 
    
    buffer = Symbol(buff, :buff)
    fromfield = Symbol(from, :field)
    tofield = Symbol(to, :field)

    transpose_name = string(transpose!)
    to_name = string(to)
    from_name = string(from)

    pack_buffer_name = string(pack_buffer!)
    unpack_buffer_name = string(unpack_buffer!)

    @eval begin
        function $transpose!(pf::TransposableArrays)
            $pack_buffer!(pf.$buffer, pf.$fromfield) # pack the one-dimensional buffer for Alltoallv! call
            sync_device!(device(pf.$fromfield)) # Device needs to be synched with host before MPI call
            Alltoallv!(VBuffer(pf.$buffer.send, pf.counts.$buff), VBuffer(pf.$buffer.recv, pf.counts.$buff), pf.comms.$buff) # Actually transpose!
            $unpack_buffer!(pf.$tofield, pf.$fromfield, pf.$buffer) # unpack the one-dimensional buffer into the 3D field
            return nothing
        end
    end
end
