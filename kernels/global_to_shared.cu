#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/barrier>

#include <Common.hh>
#include <Types.hh>
#include <cudaHelpers.cuh>
#include <global_to_shared.hh>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

__device__ void computeElem(types::f32* value, uint64_t numFlops) {
  uint32_t tmp = *value;

  for (uint32_t i = 0; i < numFlops; i++)
    tmp += tmp;
  /* Write back to avoid compiler optimization. */
  *value = tmp;
}

/**
 * performOps - execute some number of operations on each value in a tile
 *
 * Execues @numOps operations for each element in @tile, and writes it back.
 *
 * @tile: a shared memory tile
 * @numOps: the number of operations to perform for each value
 */
template <uint32_t NumElems>
__device__ void computeOnTile(types::f32 tile[NumElems], uint64_t numFlops) {
  for (uint32_t i = 0; i < NumElems; i += blockDim.x) {
    uint32_t idx = threadIdx.x + i;
    if (idx < NumElems) {
      types::f32* ptr = &tile[i + threadIdx.x];
      computeElem(ptr, numFlops);
    }
  }
}

template <uint32_t TileSize>
__global__ void globalToSharedMemSync(types::f32* globalBuffer, uint32_t globalBufferSize, uint64_t numFlops) {
  __shared__ types::f32 tile[TileSize / sizeof(types::f32)];

  /* Every block reads a partition of the global buffer. */
  uint32_t totalTiles = globalBufferSize / ARRAY_SIZE(tile);
  uint32_t tilesPerBlock = (totalTiles + gridDim.x - 1) / gridDim.x;
  uint32_t myStartTile = blockIdx.x * tilesPerBlock;
  uint32_t myEndTile = min(myStartTile + tilesPerBlock, totalTiles);

  for (uint32_t tileIdx = myStartTile; tileIdx < myEndTile; tileIdx++) {
    /* Load data from DRAM into shared memory synchronously. */
    for (uint32_t i = 0; i < ARRAY_SIZE(tile); i += blockDim.x) {
      uint32_t idx = threadIdx.x + i;
      if (idx < ARRAY_SIZE(tile))
        tile[idx] = globalBuffer[tileIdx * ARRAY_SIZE(tile) + threadIdx.x + i];
    }
    __syncthreads();
    /* Compute using the data. */
    computeOnTile<TileSize / sizeof(types::f32)>(tile, numFlops);
    __syncthreads();
  }
}

/**
 * globalToSharedMemAsyncDoubleBuffered - bench global to shared bandwidth
 *
 * NOTE: we silence the warning emitted by dynamically initializing a barrier
 * in shared memory. This is a requirement for the program to build as we build
 * with -Werror. This is a documented workaround as per the CUDA documentation
 * for barriers: https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/barrier/init.html#libcudacxx-extended-api-synchronization-barrier-barrier-init
 */
#pragma nv_diag_suppress static_var_with_dynamic_init
template <uint32_t TileSize>
__global__ void globalToSharedMemAsyncDoubleBuffered(types::f32* globalBuffer, uint32_t globalBufferSize,
                                                     uint64_t numFlops) {
  __shared__ types::f32 tiles[2][TileSize / sizeof(types::f32)];
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

  /* Calculate this block's partition. */
  const auto tileElems = ARRAY_SIZE(tiles[0]);
  uint32_t totalTiles = globalBufferSize / tileElems;
  uint32_t tilesPerBlock = (totalTiles + gridDim.x - 1) / gridDim.x;
  uint32_t myStartTile = blockIdx.x * tilesPerBlock;
  uint32_t myEndTile = min(myStartTile + tilesPerBlock, totalTiles);

  if (threadIdx.x == 0)
    init(&barrier, blockDim.x);
  __syncthreads();

  /* Early exit if this block has no tiles. */
  if (myStartTile >= myEndTile)
    return;

  auto group = cooperative_groups::this_thread_block();

  /* Load first tile for this block's partition. */
  cuda::memcpy_async(group, tiles[0], &globalBuffer[myStartTile * tileElems], sizeof(tiles[0]), barrier);
  barrier.arrive_and_wait();

  for (uint32_t tileIdx = myStartTile + 1; tileIdx < myEndTile; tileIdx++) {
    int readIdx = (tileIdx - myStartTile - 1) % 2;
    int writeIdx = (tileIdx - myStartTile) % 2;

    /* Issue the next memory copy asynchronously from this block's partition */
    cuda::memcpy_async(group, tiles[writeIdx], &globalBuffer[tileIdx * tileElems], sizeof(tiles[0]), barrier);
    /* Overlap the async copy with computation */
    computeOnTile<TileSize / sizeof(types::f32)>(tiles[readIdx], numFlops);
    barrier.arrive_and_wait();
  }

  /* Process last tile. */
  computeOnTile<TileSize / sizeof(types::f32)>(tiles[(myEndTile - myStartTile - 1) % 2], numFlops);
}

template <uint32_t TileSize>
using globalToSharedMemKernel = void (*)(types::f32*, uint32_t, uint64_t);

template <uint32_t TileSize>
static globalToSharedMemKernel<TileSize> getKernel(globalToShared::mode mode) {
  switch (mode) {
    case globalToShared::SYNC:
      return globalToSharedMemSync<TileSize>;
    case globalToShared::ASYNC_2X_BUFFERED:
      return globalToSharedMemAsyncDoubleBuffered<TileSize>;
    case globalToShared::ASYNC_4X_BUFFERED:
      throw std::runtime_error("async 4x buffered not implemented");
    default:
      throw std::invalid_argument("invalid mode");
  }
}

/**
 * launchGlobalToSharedKernel - launch a global-to-shared memory bench kernel
 *
 * @return the number of milliseconds that the kernel took to execute.
 *
 * @mode: the mode, e.g., synchronous, double-buffered, quadruble buffered.
 * @globalbuffer: a large memory region with arbitrary data that is read into
 *                shared memory by the GPU threads.
 * @numOps: the number of (integer) operations to perform per loaded tile value.
 * @threadsPerBlock: the number of threads per block.
 * @numBlocks: the number of thread blocks.
 */
template <uint32_t TileSize>
float launchGlobalToSharedKernel(globalToShared::mode mode, const std::vector<types::f32>& globalBuffer,
                                 uint64_t numFlops, uint64_t threadsPerBlock, uint64_t numBlocks) {
  types::f32* dGlobalBuffer;

  throwOnErr(cudaMalloc(&dGlobalBuffer, globalBuffer.size() * sizeof(types::f32)));
  throwOnErr(
      cudaMemcpy(dGlobalBuffer, globalBuffer.data(), globalBuffer.size() * sizeof(types::f32), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto kernel = getKernel<TileSize>(mode);

  cudaEventRecord(start);

  kernel<<<static_cast<unsigned int>(numBlocks), static_cast<unsigned int>(threadsPerBlock)>>>(
      dGlobalBuffer, static_cast<uint32_t>(globalBuffer.size()), numFlops);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  throwOnErr(cudaDeviceSynchronize());
  throwOnErr(cudaGetLastError());
  throwOnErr(cudaFree(dGlobalBuffer));

  return milliseconds;
}

template float launchGlobalToSharedKernel<common::KiB>(globalToShared::mode mode,
                                                       const std::vector<types::f32>& globalBuffer, uint64_t numFlops,
                                                       uint64_t threadsPerBlock, uint64_t numBlocks);

template float launchGlobalToSharedKernel<2 * common::KiB>(globalToShared::mode mode,
                                                           const std::vector<types::f32>& globalBuffer,
                                                           uint64_t numFlops, uint64_t threadsPerBlock,
                                                           uint64_t numBlocks);

template float launchGlobalToSharedKernel<4 * common::KiB>(globalToShared::mode mode,
                                                           const std::vector<types::f32>& globalBuffer,
                                                           uint64_t numFlops, uint64_t threadsPerBlock,
                                                           uint64_t numBlocks);
