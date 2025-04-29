#include "cupim.h"
#include <cuda_runtime.h>

struct chunk {
	pimBuffer_t<int> rowA;
	pimBuffer_t<int> vec;
	int output;
};

__global__ void gemvCUPIM(chunk* plan, int width) {
	plan->output = 0;
	for (int j = 0; j < width; j++) {
		plan->output += plan->rowA[j] * plan->vec[j];
	}
}

int main() {
	int width = 1024;
	int height = 1024;

	int *mat = (int*) malloc(sizeof(int) * width * height);
	int *vec = (int*) malloc(sizeof(int) * width);
	int *out = (int*) malloc(sizeof(int) * height);

	// define chunk dimensions
	dim3 chunk_size(1, 1, 1);
	dim3 num_chunks(width, height, 1);

	// define chunk
	chunkId_t i = num_chunks.x;
	chunkId_t j = num_chunks.y;

	chunk plan;
	plan.rowA = Row(mat, i);
	plan.vec = Col(vec, 0);

	int input_size_per_chunk = sizeof(int) * (width + width);
	int output_size_per_chunk = sizeof(int);
	int size_per_chunk = input_size_per_chunk + output_size_per_chunk;

	int size_across_all_cores = size_per_chunk * num_chunks.x;
	
	chunk* plan_d;
	pimMalloc(plan_d, size_across_all_cores);

	pimMemCopyPlanToDevice(&plan, plan_d, size_across_all_cores);
	gemvCUPIM<<<num_chunks, chunk_size>>>(plan_d, width);
	pimMemCopyPlanToHost(&plan, plan_d, size_across_all_cores);

	pimLoadFromPlan(out, plan_d->output, sizeof(int) * height);

	pimFree(plan_d);
	free(mat);
	free(vec);
}
