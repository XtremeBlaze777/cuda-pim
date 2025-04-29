#include "Halide.h"
using namespace Halide;

class GEMV : public Generator<GEMV> {
public:
    Input<Buffer<int32_t>> rowA{ "rowA", 1 };
    Input<Buffer<int32_t>> vec{ "vec", 1 };
	Input<int32_t> width{ "width" };

    Output<Buffer<int32_t>> output{ "output", 0 };

    void generate() {
		RDom j(0, width, "j");
        output() = sum(rowA(j) * vec(j));
    }
};

HALIDE_REGISTER_GENERATOR(GEMV, gemv)

class GEMVFull : public Generator<GEMVFull> {
public:
    Input<Buffer<int32_t>> A{ "A", 2 };
    Input<Buffer<int32_t>> vec{ "vec", 1 };
	Input<int32_t> width{ "width" };

    Output<Buffer<int32_t>> output{ "output", 1 };

    void generate() {
		RDom i(0, width, "i");
        output(j) = sum(A(i, j) * vec(i));
    }
private:
	Var j{ "j" };
};

HALIDE_REGISTER_GENERATOR(GEMVFull, gemvFull)
