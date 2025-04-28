#include "Halide.h"
using namespace Halide;

#define TILE_SIZE 4

class MatMul : public Generator<MatMul> {
public:
    // Inputs
    Input<Buffer<float>> A{"A", 2}; // 2D matrix
    Input<Buffer<float>> B{"B", 2}; // 2D matrix
    Input<int> K{"K"};              // Shared dimension size

    // Outputs
    Output<Func> C{"C"};

    // The function
    void generate() {
        Var i("i"), j("j");
        RDom r(0, K);
		
        C(i, j) = 0.0f;
        C(i, j) += A(i, r) * B(r, j);
    }
	
	void schedule() {
		Var i("i"), j("j"), io("io"), ii("ii"), jo("jo"), jj("jj");

		// Schedule pure definition (C(i, j) = 0)
		C.bound(i, 0, A.dim(0).extent()) // Optional: Bound for safety
			.bound(j, 0, B.dim(1).extent());

		// Schedule update (C(i, j) += ...)
		C.update()
			.tile(i, j, io, jo, ii, jj, TILE_SIZE, TILE_SIZE)
			// .parallel(io)
			// .vectorize(ii, 8)
			// .unroll(jj)
            ;
	}
	
};

HALIDE_REGISTER_GENERATOR(MatMul, matmul)
