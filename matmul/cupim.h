#include <vector>
#include <string>
#include <stdexcept>

#define chunkId_t int
#define pimStatus_t int

#define pimBuffer_t std::vector
#define pimChunkBuffer_t(type) pimBuffer_t<pimBuffer_t<type>>

#ifndef CHUNK_TYPE
#define CHUNK_TYPE
struct chunk {
	pimChunkBuffer_t(int) rows;
	pimChunkBuffer_t(int) cols;
	pimChunkBuffer_t(int) output;
};
#endif

#define DEVICE_NOT_INITIALIZED -1
#define PIM_SUCCESS 0
#define PIM_LOGIC_ERROR 1

static int pimErrno = DEVICE_NOT_INITIALIZED;

pimStatus_t pimAlloc(void* buffer, int size);

pimStatus_t pimMemCopyHostToDevice(void* src, void* dst, int size);
pimStatus_t pimMemCopyDeviceToHost(void* src, void* dst, int size);

pimChunkBuffer_t(int) Row(int* A, int i);
pimChunkBuffer_t(int) Col(int* B, int j);

void pimFree(void* buffer);

int pimGetUnitCount();
bool pimPrimitivesAvailable();

chunk* pimPlanAllocHost(int size);
chunk* pimPlanAllocDevice(chunk* plan, int size);

pimStatus_t pimLoadPlanFromDevice(void* buffer, chunk* plan, int size);

pimStatus_t pimDeviceSynchronize();

pimStatus_t pimMAC(int* dst, pimBuffer_t<int> src1, pimBuffer_t<int> src2);

std::string pimGetErrorString(pimStatus_t status);
void pimError(pimStatus_t status) {
	pimErrno = status;
}

