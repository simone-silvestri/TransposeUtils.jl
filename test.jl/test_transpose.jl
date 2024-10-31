using MPI
using TransposeUtils
using Test

MPI.Init()

ranks_x = 2
ranks_y = 2

@testset "Test the transpose" begin
    partition = Partition(CPU(); ranks = (ranks_x, ranks_y, 1))
    carray    = zeros(partition, ComplexF64, 200, 200, 100)
    
    set!(carray, (i, j, k) -> rand() + rand()im)
    
    tarray = TransposableArrays(carray, partition)

    transpose_z_to_y!(tarray)
    transpose_y_to_x!(tarray)
    transpose_x_to_y!(tarray)
    transpose_y_to_z!(tarray)

    @test tarray.zfield == carray

    # Test the transform
    transform = DistributedTransform(tarray)

    # After a forward and backward transform the result should not have changed
    fft!(transform)
    ifft!(transform)

    # The result should be the same
    tarray.zfield ≈ carray
end