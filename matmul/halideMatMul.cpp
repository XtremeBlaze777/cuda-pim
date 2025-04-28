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

        Expr io = i / TILE_SIZE;
        Expr ii = i % TILE_SIZE;
        Expr jo = j / TILE_SIZE;
        Expr jj = j % TILE_SIZE;

        RDom r(0, K, "r");
		
        Expr expr1 = A(io * TILE_SIZE + ii, r);
        Expr expr2 = B(r, jo * TILE_SIZE + jj);
        Expr expr3 = sum(expr1 * expr2);

        C(i, j) = expr3;
    }
	
	void schedule() {
		Var i("i"), j("j");

		// Schedule pure definition (C(i, j) = 0)
		C.bound(i, 0, A.dim(0).extent()) // Optional: Bound for safety
			.bound(j, 0, B.dim(1).extent());

		// Schedule update (C(i, j) += ...)
		// C.update()
			// .tile(i, j, io, jo, ii, jj, TILE_SIZE, TILE_SIZE)
			// .parallel(io)
			// .vectorize(ii, 8)
			// .unroll(jj)
            // ;
	}
	
};

HALIDE_REGISTER_GENERATOR(MatMul, matmul)
