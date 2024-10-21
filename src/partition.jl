import Base

#####
##### Partitioning
#####

struct Partition{A, R, I, C} 
    device :: A
    ranks :: R
    local_index :: I
    connectivity :: C
end

#####
##### Constructors
#####

function Partition(device = CPU(); 
                   ranks  = (1, 1, 1))

    if !(MPI.Initialized())
        @info "MPI has not been initialized, so we are calling MPI.Init()."
        MPI.Init()
    end

    if prod(ranks) != MPI.Comm_size(MPI.COMM_WORLD)
        throw(ArgumentError("The number of MPI ranks does not match the product of the ranks in the partition."))
    end

    communicator = MPI.COMM_WORLD
    mpi_ranks    = MPI.Comm_size(communicator)

    if isnothing(ranks) # default partition
        ranks = mpi_ranks
    end

    Rx, Ry, Rz  = ranks
    local_rank  = MPI.Comm_rank(communicator)
    local_index = rank2index(local_rank, Rx, Ry, Rz)

    # The rank connectivity _ALWAYS_ wraps around (The cartesian processor "grid" is `Periodic`)
    local_connectivity = RankConnectivity(local_index, ranks) 

    # Assign CUDA device if on GPUs
    if device isa GPU
        local_comm = MPI.Comm_split_type(communicator, MPI.COMM_TYPE_SHARED, local_rank)
        node_rank  = MPI.Comm_rank(local_comm)
        device!(node_rank % ndevices()) 
    end

    return Partition(device,
                     ranks,
                     local_index,
                     local_connectivity)
end

@inline device(p::Partition) = p.device

#####
##### Converting between index and MPI rank taking k as the fast index
#####

index2rank(i, j, k, Rx, Ry, Rz) = (i-1)*Ry*Rz + (j-1)*Rz + (k-1)

function rank2index(r, Rx, Ry, Rz)
    i = div(r, Ry*Rz)
    r -= i*Ry*Rz
    j = div(r, Rz)
    k = mod(r, Rz)
    return i+1, j+1, k+1  # 1-based Julia
end

@inline ranks(p::Partition) = p.ranks

#####
##### Rank connectivity graph
#####

struct RankConnectivity{E, W, N, S, SW, SE, NW, NE}
         east :: E
         west :: W
        north :: N
        south :: S
    southwest :: SW
    southeast :: SE
    northwest :: NW
    northeast :: NE
end

const NoConnectivity = RankConnectivity{Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing}

"""
    RankConnectivity(; east, west, north, south, southwest, southeast, northwest, northeast)

Generate a `RankConnectivity` object that holds the MPI ranks of the neighboring processors.
"""
RankConnectivity(; east, west, north, south, southwest, southeast, northwest, northeast) =
    RankConnectivity(east, west, north, south, southwest, southeast, northwest, northeast)

# The "Periodic" topologies are `Periodic`, `FullyConnected` and `RightConnected`
# The "Bounded" topologies are `Bounded` and `LeftConnected`
function increment_index(i, R)
    R == 1 && return nothing
    if i+1 > R
        return 1
    else
        return i+1
    end
end

function decrement_index(i, R)
    R == 1 && return nothing
    if i-1 < 1
        return R
    else
        return i-1
    end
end

function RankConnectivity(local_index, ranks)
    i, j, k = local_index
    Rx, Ry, Rz = ranks

    i_east  = increment_index(i, Rx)
    i_west  = decrement_index(i, Rx)
    j_north = increment_index(j, Ry)
    j_south = decrement_index(j, Ry)

     east_rank = isnothing(i_east)  ? nothing : index2rank(i_east,  j, k, Rx, Ry, Rz)
     west_rank = isnothing(i_west)  ? nothing : index2rank(i_west,  j, k, Rx, Ry, Rz)
    north_rank = isnothing(j_north) ? nothing : index2rank(i, j_north, k, Rx, Ry, Rz)
    south_rank = isnothing(j_south) ? nothing : index2rank(i, j_south, k, Rx, Ry, Rz)

    northeast_rank = isnothing(i_east) || isnothing(j_north) ? nothing : index2rank(i_east, j_north, k, Rx, Ry, Rz)
    northwest_rank = isnothing(i_west) || isnothing(j_north) ? nothing : index2rank(i_west, j_north, k, Rx, Ry, Rz)
    southeast_rank = isnothing(i_east) || isnothing(j_south) ? nothing : index2rank(i_east, j_south, k, Rx, Ry, Rz)
    southwest_rank = isnothing(i_west) || isnothing(j_south) ? nothing : index2rank(i_west, j_south, k, Rx, Ry, Rz)

    return RankConnectivity(west=west_rank, east=east_rank,
                            south=south_rank, north=north_rank,
                            southwest=southwest_rank,
                            southeast=southeast_rank,
                            northwest=northwest_rank,
                            northeast=northeast_rank)
end

"""
    concatenate_local_sizes(local_size, arch::Distributed) 

Return a 3-Tuple containing a vector of `size(array, dim)` for each rank in 
all 3 directions.
"""
concatenate_local_sizes(local_size, p::Partition) = 
    Tuple(concatenate_local_sizes(local_size, p, d) for d in 1:length(local_size))

concatenate_local_sizes(sz, p, dim) = concatenate_local_sizes(sz[dim], p, dim)

function concatenate_local_sizes(n::Number, p::Partition, dim)
    R = p.ranks[dim]
    r = p.local_index[dim]
    N = zeros(Int, R)

    r1, r2 = p.local_index[[1, 2, 3] .!= dim]
    
    if r1 == 1 && r2 == 1
        N[r] = n
    end

    MPI.Allreduce!(N, +, MPI.COMM_WORLD)
    
    return N
end