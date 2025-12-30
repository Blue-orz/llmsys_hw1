
* - index_to_position：将多维数组的索引转换为紧凑一维数组中的内存位置（偏移量）
* - to_index：将紧凑一维数组中的内存位置转换为多维数组的索引
* - broadcast_index：将小尺寸数组的索引转换为大尺寸数组的索引（用于广播机制）

__device__ int index_to_position(const int* index, const int* strides, int num_dims)
__device__ void to_index(int ordinal, const int* shape, int* out_index, int num_dims)
__device__ void broadcast_index(const int* big_index, const int* big_shape, const int* shape, int* out_index, int num_dims_big, int num_dims) 

float* out,
int* out_shape,
int* out_strides,

int out_size,

float* a_storage,
int* a_shape,
int* a_strides,

int reduce_dim,

float reduce_value,

int shape_size,
int fn_id


/**
   * Multiply two (compact) matrices into an output (also comapct) matrix. Matrix a and b are both in a batch
   * format, with shape [batch_size, m, n], [batch_size, n, p].
   * Requirements:
   * - All data must be first moved to shared memory.
   * - Only read each cell in a and b once.
   * - Only write to global memory once per kernel.
   * There is guarantee that a_shape[0] == b_shape[0], a_shape[2] == b_shape[1],
   * and out_shape[0] == a_shape[0], out_shape[1] == b_shape[1]
   *
   * Args:
   *   out: compact 1D array of size batch_size x m x p to write the output to
   *   out_shape: shape of the output array
   *   out_strides: strides of the output array
   *   a_storage: compact 1D array of size batch_size x m x n
   *   a_shape: shape of the a array
   *   a_strides: strides of the a array
   *   b_storage: comapct 2D array of size batch_size x n x p
   *   b_shape: shape of the b array
   *   b_strides: strides of the b array
   *
   * Returns:
   *   None (Fills in out array)
   */