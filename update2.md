- Full Generator:
    ```c
    // This is Halide code
    class TensorAdd : public Generator<TensorAdd> {
    public:
        Input<Buffer<int32_t>> A{ "A", 2 };
        Input<Buffer<int32_t>> B{ "B", 2 };
        Output<Buffer<int32_t>> output{ "output", 2 };

        void generate() {
            output(x, y) = A(x,y) + B(x,y);
        }
    private:
        Var x{ "x" }, y{ "y" };
    };

    HALIDE_REGISTER_GENERATOR(TensorAdd, tensor_add)
    ```

    Corresponding pim-cuda code:
    ```c
    __pim__ TensorAdd(Buffer* A, Buffer* B) {
        // By using threadIdx, we don't need to explicitly declare Var x,y
        output[threadIdx.x][threadIdx.y] = A[threadIdx.x][ThreadIdx.y] + B[ThreadIdx.x][ThreadIdx.y];
    }

    int main() {
        buff *A, *B, *output_buff;
        int output_dim = 2; // this is what gives us threadIdx.x(y)

        tensor_add<<<output_buff, output_dim>>>(A, B);
    }

    ```

<br><br>

- Rdom is basically inner loops, so we will allow loops in our PIM kernels and the compiler will convert those into Rdom like this matmul code:
    ```c
    Var x = threadIdx.x;
    Var y = threadIdx.y;
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += A[i][y] * B[x][i];
    }
    output[x][y] = sum;
    ```
    would get translated to halide:
    ```c
    sum(x,y) = 0;
    RDom i(0, N);  // this is the inc var from the loop
    output(x, y) += A(i, y) * B(x, i);
    ```

- Halide has select statements that can selectively choose an expression to update the halide Func:
    ```c
    if (x%2 == 0) {
        output[x][y] = x + y;
    }
    else if (x%3 == 0) {
        output[x][y] = x * y;
    }
    else {
        output[x][y] = x + 15;
    }
    ```
    would become:
    ```c
    Func f1, f2, f3, f4;

    Expr cond1 = x%2 == 0;
    f1(x,y) = x + y;

    Expr cond2 = x%3 == 0;
    f2(x,y) = x * y;

    f3(x,y) = x + 15;

    f4(x,y) = select(cond2, f2(x,y), f3(x,y));
    output(x,y) = select(cond1, f1(x,y), f4(x,y));
    ```

<br><br>

- Building generators:
    - We need the main from [GenGen.cpp](https://github.com/halide/Halide/blob/main/tools/GenGen.cpp) to generate either c source code or static libraries with our generate generators
    - Just compile with that file (and the other Halide flags) and it will produce a binary that when run will actually produce the code
    - If the binary can't find `libHalide.so`, provide the path to the lib folder as a value to rpath when compiling: `-Wl,-rpath=~/.local/lib/python3.8/site-packages/halide/lib64/`
    - `./a.out -g tensor_add -o . -e c_header,c_source,static_library target=host`
    - and now you should see the generator source in the folder specified with -o