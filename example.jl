using MPI
using TransposeUtils
using BenchmarkTools
using FFTW

MPI.Init()

ranks_x = 2
ranks_y = 2

partition = Partition(CPU(); ranks = (ranks_x, ranks_y, 1))
carray = zeros(partition, ComplexF64, 20, 20, 10)
@distributed_info size(carray)

set!(carray, (i, j, k) -> rand() + rand()im)

# Container with data structures useful to perform the transposition
# the data is contained in
# - tarray.xfield, the x-local configuration (rank size in x == 1)
# - tarray.yfield, the y-local configuration (rank size in y == 1)
# - tarray.zfield, the z-local configuration (rank size in z == 1, the default)
tarray = TransposableArrays(carray, partition)
@distributed_info size(tarray.xfield)
@distributed_info size(tarray.yfield)
@distributed_info size(tarray.zfield)

# Transposing between the different configurations
# z-local (the default), y-local, and x-local
transpose_z_to_y!(tarray)
transpose_y_to_x!(tarray)
transpose_x_to_y!(tarray)
transpose_y_to_z!(tarray)

# At the end of the transposition cycle the result should 
# not have changed
@distributed_info tarray.zfield == carray

# Test the transform
transform = DistributedTransform(tarray)

# After a forward and backward transform the result should not have changed
fft(transform)
ifft(transform)

@distributed_info tarray.zfield ≈ carray