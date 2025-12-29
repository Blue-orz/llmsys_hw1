
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
int out_shape_size,

float* a_storage,
int* a_shape,
int* a_strides,
int a_shape_size,

float* b_storage, 
int* b_shape, 
int* b_strides,
int b_shape_size,

int fn_id