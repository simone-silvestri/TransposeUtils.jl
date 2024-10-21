struct TransposableArrays{FX, FY, FZ, YZ, XY, C, Comms}
    xfield :: FX # X-direction is free (x-local)
    yfield :: FY # Y-direction is free (y-local)
    zfield :: FZ # Z-direction is free (original field, z-local)
    yzbuff :: YZ # if `nothing` slab decomposition with `Ry == 1`
    xybuff :: XY # if `nothing` slab decomposition with `Rx == 1`
    counts :: C
    comms  :: Comms
end

const SlabYArrays = TransposableArrays{<:Any, <:Any, <:Any, <:Nothing} # Y-direction is free
const SlabXArrays = TransposableArrays{<:Any, <:Any, <:Any, <:Any, <:Nothing} # X-direction is free

function TransposableArrays(array_in, zarch, FT = eltype(array_in))

    zsize = size(array_in)
    ysize, yarch = twin_configuration(zsize, zarch, eltype(array_in); local_direction = :y)
    xsize, xarch = twin_configuration(zsize, zarch, eltype(array_in); local_direction = :x)

    xN = xsize
    yN = ysize
    zN = zsize

    Rx, Ry, _ = zarch.ranks

    z_array = deepcopy(array_in)
    y_array = Ry == 1 ? z_array : zeros(device(yarch), FT, yN...)
    x_array = Rx == 1 ? y_array : zeros(device(xarch), FT, xN...)

    # One dimensional buffers to "pack" three-dimensional data in for communication 
    yzbuffer = Ry == 1 ? nothing : (send = zeros(device(zarch), FT, prod(yN)), 
                                    recv = zeros(device(zarch), FT, prod(zN)))
    xybuffer = Rx == 1 ? nothing : (send = zeros(device(zarch), FT, prod(xN)), 
                                    recv = zeros(device(zarch), FT, prod(yN)))

    yzcomm = MPI.Comm_split(MPI.COMM_WORLD, zarch.local_index[1], zarch.local_index[1])
    xycomm = MPI.Comm_split(MPI.COMM_WORLD, yarch.local_index[3], yarch.local_index[3])

    zRx, zRy, zRz = ranks(zarch) 
    yRx, yRy, yRz = ranks(yarch) 

    # size of the chunks in the buffers to be sent and received
    # (see the docstring for the `transpose` algorithms)    
    yzcounts = zeros(Int, zRy * zRz)
    xycounts = zeros(Int, yRx * yRy)

    yzrank = MPI.Comm_rank(yzcomm)
    xyrank = MPI.Comm_rank(xycomm)

    yzcounts[yzrank + 1] = yN[1] * zN[2] * yN[3]
    xycounts[xyrank + 1] = yN[1] * xN[2] * xN[3]

    MPI.Allreduce!(yzcounts, +, yzcomm)
    MPI.Allreduce!(xycounts, +, xycomm)

    return TransposableArrays(x_array, y_array, z_array, 
                              yzbuffer, xybuffer,
                              (; yz = yzcounts, xy = xycounts),
                              (; yz = yzcomm,   xy = xycomm))
end

function twin_configuration(zsize, zarch, FT::DataType = Float64; local_direction = :y)

    ri, rj, rk = zarch.local_index

    R = zarch.ranks

    nx, ny, nz = zsize
    Nx, Ny, Nz = global_size(zarch, zsize)
    
    child_device = device(zarch)

    if local_direction == :y
        ranks = R[1], 1, R[2]

        nnx, nny, nnz = nx, Ny, nz ÷ ranks[3]

        if (nnz * ranks[3] < Nz) && (rj == ranks[3])
            nnz = Nz - nnz * (ranks[3] - 1)
        end
    elseif local_direction == :x
        ranks = 1, R[1], R[2]

        nnx, nny, nnz = Nx, Ny ÷ ranks[2], nz ÷ ranks[3]

        if (nny * ranks[2] < Ny) && (ri == ranks[2])
            nny = Ny - nny * (ranks[2] - 1)
        end
    elseif local_direction == :z
        #TODO: a warning here?
        return grid
    end

    new_arch = Partition(child_device; ranks)
    new_size = (nnx, nny, nnz)

    return new_size, new_arch
end
