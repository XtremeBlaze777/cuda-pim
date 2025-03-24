//initial vecadd tests

int main(){
    //CPU SIDED CODE
    // cudaMalloc(A);
    // cudaMalloc(B);
    // cudaMalloc(c);

    halideOutput(c)

    tensorAdd<<<1, 1>>>(output, inputs);
}
/*
function arguments (output data, input data...) 
example  tensorAdd(int *output, int *a, int*b)
alternatively tensorAdd(int *output, argv)

The goal here is to allow the compiler to easily know the ouput to realize
*/
//or let compiler figure it out
__memory__ tensorAdd(int *c, int *A, int *B){ 
    //goal to generate code similar to halideTensorAdd.cpp
    int threadId = blockdim.x * blockIdx.x + threadIdx.x; //cuda style thread indexing (subject to change)

    c[threadId] = A[threadId] + B[threadId]
}

// decorator is an interesting idea but is against the goal of CUDA-like
@halide_generator(COMPUTE_ROOT | REORDER | VECTORIZE)
tensorAdd(int *c, int *A, int *B, dim2 length){ 
    c(x,y) = A(x,y) + B(x,y);
}