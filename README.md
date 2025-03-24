- because the recommended install for halide is `pip install halide`, the .so gets put in .local/site-packages, thus we need to manually tell the compiler where it's located (adjust python version accordingly):
    ```c
    export HALIDE_FLAGS="-g -I $HOME/.local/lib/python3.11/site-packages/halide/include/ -L $HOME/.local/lib/python3.11/site-packages/halide/lib64/ -lHalide -lpthread -ldl -std=c++17"
    alias halide="LD_LIBRARY_PATH=$HOME/.local/lib/python3.11/site-packages/halide/lib64"
    ```
    and so the run instructions become:
    ```bash
    g++ file.cpp $HALIDE_FLAGS
    halide ./a.out
    ```

- the point of halide is that it has a distinction between the algorithm and the pipeline scheduling (unrolling, vectorize, etc). CUDA has macros defined such as `#define cudaMemcpyDeviceToHost 1` which is passed in as the last argument of `cudaMemcpy`; so an idea is to have a bitmask of various halide scheduling options and pass those in as the last argument. This is kind of how we gave permissions to file descriptors when we were doing C systems programming for the OS class.
    ```c
    // header file
    #define COMPUTE_ROOT 1
    #define REORDER 0b10
    #define VECTORIZE 0b100

    // client code
    output(..., COMPUTE_ROOT | REORDER | VECTORIZE);
    ```

- most Halide "primitives" pass the symbol name as a string (`Var x{"x"}`) so that there is debug info (c++ doesn't have `__name__`). This can be a feature of the compiler/transpiler rather than the client code having to type it. For example, the halide emitter could insert a string of the symbol name as a parameter.

- Halide also lets you specify buffers as input or output which is how it constructs it's function signature:
    ```c
    Input<Buffer<int32_t>> A{ "A", 2 };
    Input<Buffer<int32_t>> B{ "B", 2 };
    Output<Buffer<int32_t>> output{ "output", 2 };

    void genererate() {
        output(x, y) = A(x, y) + B(x, y);
    }
    ```
    so there are a couple of approaches we could try:
    - C++ has variadic functions which are similar to python's `*args` and `**kwargs`: `printf(char* str, ...)`
    - Another option is to just use an `argc, argv` approach where we just pass an array of inputs and an array of outputs
    - We could also just offload this to the compiler which could be tricky but it can recgonize what is on the left and right-hand side of the expression.

- Understanding the LLVM IR representation of TensorAdd (`llvmTensorAdd.ll`):
    - we are passing in %arg and %arg.1 which must be the input buffers
    - there is a difference between pimAlloc and alloca, it seems like pimAlloc specifically designates it as a buffer for the compute units on the DIMM, while alloca will just get it to RAM.