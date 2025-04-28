
typedef struct chunk{
    int myRowA;
    int myColB;
    //need to signify result for Halide? working on result still
   // res *chunk_result;
}plan


int main(){
    //generate data
    int width = 512;
    int *h_data = malloc(sizeof(int) * width * width);
    int *h_res = malloc(sizeof(int) * width * width);
    for(int i = 0; i < width * width; i++){
        h_data[i] = //some data
    }


    //alloc space for PIM
    int *p_data;
    int *p_result;
    int size_across_all_cores = sizeof(int) * ((width + width) * pimCount * batches);

    int result_size = sizeof(int) * (width * width);

    pimAlloc(&p_data, size_across_all_cores);

    //figuring out res still
    pimAlloc(&p_result, result_size);



    //define chunks
    dim3 chunk_size(1, 1, 1);
    dim3 chunk_grid(width, width, 1);

    //define chunk
    chunkId_t i = chunk_grid.x
    chunkId_t j = chunk_grid.y


    plan chunk_plan;
    chunk_plan.myRowA = Row(A, i);
    chunk_plan.myColB = Col(B, j);

    /* if multi rows
    //ranges of rows or mulitple orws
    chunk_plan.myRowsA = [Row(A, 2 * i), Row(A, 2 * i + 1)]

    chunk_plan.myRowsA = RowRange(A, tile_width * i, i + tile_width)
    //or something like that divides rows into slices
    //
    
    */

    pimMemCopyPlan(p_data, chunk_plan, );

    kernel<<<chunk_size, chunk_grid>>>(chunk_plan, p_result);
    //copy data to PIM
    //WIP
    //if result ? row major
    dim3 chunk_res_size(1,1,1)
    pimMemCopy(h_res, p_data, result_size, )


    return 0;
}

__PIM__ naive_kernel(){
    int* myRow = chunk.myRowA;
    int* myCol = chunk.mycolB;

// rest of this
    


}