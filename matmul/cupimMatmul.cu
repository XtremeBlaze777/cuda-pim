#include "cupim.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_WIDTH 4

#ifndef CHUNK_TYPE
#define CHUNK_TYPE
struct chunk {
	pimChunkBuffer_t(int) rows;
	pimChunkBuffer_t(int) cols;
	pimChunkBuffer_t(int) output;
};
#endif

__global__ void matmulTiled(chunk* plan) {
	// each plan contains the rows and columns necessary for the tile in C
	for (int i = 0; i < plan->rows.size(); i++) {
		for (int j = 0; j < plan->cols.size(); j++) {
			plan->output[i][j] = 0;

			pimBuffer_t<int> cur_row = plan->rows[i];
			pimBuffer_t<int> cur_col = plan->cols[j];
			if (cur_row.size() != cur_col.size()) {
				fprintf(stderr, "Error: row and column sizes do not match\n");
				return;
			}
			for (int k = 0; k < cur_row.size(); k++) {
				plan->output[i][j] += cur_row[k] * cur_col[k];
			}
		}
	}
}

int main() {
	int mat_width = 1024;

	int *A = (int*) malloc(sizeof(int) * mat_width * mat_width);
	int *B = (int*) malloc(sizeof(int) * mat_width * mat_width);
	int *C = (int*) malloc(sizeof(int) * mat_width * mat_width);

	//define chunk dimensions
    dim3 chunk_size(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_chunks(mat_width, mat_width, 1);

	//define chunk
    chunkId_t i = num_chunks.x;
    chunkId_t j = num_chunks.y;

	// pim size of data
	int batches = num_chunks.x * num_chunks.y;
	int batches_per_core = ceil( ((float)batches) / ((float)pimGetUnitCount()) );

	int input_size_per_batch = sizeof(int) * 2*(mat_width * mat_width);
	int output_size_per_batch = sizeof(int) * (mat_width * mat_width);

    int input_size_across_all_cores = input_size_per_batch * batches;
	int output_size_across_all_cores = output_size_per_batch * batches;

	int size_across_all_cores = input_size_across_all_cores + output_size_across_all_cores;

	// host side plan definition
	chunk* plan_host = pimPlanAllocHost(size_across_all_cores, chunk_def=chunk);  // allocates size_across/sizeof(chunk) chunks
	plan_host->rows = Row(A, i);
	plan_host->cols = Col(B, j);
	// plan_host->setRows(Row(A, i));
	
    // alloc space for PIM
	chunk* plan_device;
	pimPlanAllocDevice(plan_device, size_across_all_cores);
	pimMemCopyHostToDevice(plan_host, plan_device, sizeof(chunk));

    int result_size = sizeof(int) * (mat_width * mat_width);

	matmulTiled<<<chunk_size, num_chunks>>>(plan_device);
	pimDeviceSynchronize();

	pimLoadPlanFromDevice(C, plan_device, size_across_all_cores);

	free(A);
	free(B);
	free(C);
}