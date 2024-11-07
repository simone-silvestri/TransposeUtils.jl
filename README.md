# TransposeUtils.jl

Reproducer app for Oceananigans' distributed FFT transform

### Before running

Make sure that MPI is correctly configured:
run
```julia
julia> using Pkg

julia> Pkg.add("MPIPreferences")

julia> using MPIPreferences

julia> MPIPreferences.use_system_binary()
```

### Running the example:

login in a compute node with 4 tasks and 4 GPUs.

```
ssilvest@nid001040:/pscratch/sd/s/ssilvest/TransposeUtils.jl> srun -n 4 $JULIA --project example.jl
┌ Info: rank 0
└   size(carray) = (20, 20, 10)
┌ Info: rank 1
└   size(carray) = (20, 20, 10)
┌ Info: rank 2
└   size(carray) = (20, 20, 10)
┌ Info: rank 3
└   size(carray) = (20, 20, 10)
┌ Info: rank 0
└   tarray.zfield == carray = true
┌ Info: rank 1
└   tarray.zfield == carray = true
┌ Info: rank 2
└   tarray.zfield == carray = true
┌ Info: rank 3
└   tarray.zfield == carray = true
┌ Info: rank 0
└   tarray.zfield ≈ carray = false
┌ Info: rank 1
└   tarray.zfield ≈ carray = false
┌ Info: rank 2
└   tarray.zfield ≈ carray = false
┌ Info: rank 3
└   tarray.zfield ≈ carray = false
```

### Running the test on CPUs:

login in a compute node with 4 tasks and 4 GPUs.

CPU test:
```
ssilvest@nid001040:/pscratch/sd/s/ssilvest/TransposeUtils.jl> srun -n 4 $JULIA --project test/test_transpose.jl
[ Info: 4
[ Info: 4
[ Info: 4
[ Info: 4
┌ Info: rank 0
└   tarray.zfield ≈ carray = false
┌ Info: rank 1
└   tarray.zfield ≈ carray = false
┌ Info: rank 2
└   tarray.zfield ≈ carray = false
┌ Info: rank 3
└   tarray.zfield ≈ carray = false
Test Summary:      | Pass  Total     Time
Test Summary:      | Pass  Total     Time
Test Summary:      | Pass  Total     Time
Test Summary:      | Pass  Total     Time
Test the transpose |    1      1  2m01.7s
Test the transpose |    1      1  2m01.7s
Test the transpose |    1      1  2m01.7s
[ Info: Tests passed!
[ Info: Tests passed!
[ Info: Tests passed!
Test the transpose |    1      1  2m01.7s
[ Info: Tests passed!
```

for GPU test, pass `GPU_TEST=true` environment variable:
```
ssilvest@nid001040:/pscratch/sd/s/ssilvest/TransposeUtils.jl> GPU_TEST=true srun -n 4 $JULIA --project test/test_transpose.jl
[ Info: 4
[ Info: 4
[ Info: 4
[ Info: 4
┌ Info: rank 0
└   tarray.zfield ≈ carray = false
┌ Info: rank 1
└   tarray.zfield ≈ carray = false
┌ Info: rank 2
└   tarray.zfield ≈ carray = false
┌ Info: rank 3
└   tarray.zfield ≈ carray = false
Test Summary:      | Pass  Total     Time
Test Summary:      | Pass  Total     Time
Test Summary:      | Pass  Total     Time
Test Summary:      | Pass  Total     Time
Test the transpose |    1      1  2m01.7s
Test the transpose |    1      1  2m01.7s
Test the transpose |    1      1  2m01.7s
[ Info: Tests passed!
[ Info: Tests passed!
[ Info: Tests passed!
Test the transpose |    1      1  2m01.7s
[ Info: Tests passed!
```
