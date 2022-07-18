#ifndef TORQUE_GPU_LIB_HELPER_CUH
#define TORQUE_GPU_LIB_HELPER_CUH

#define DEBUG(x) do { printf("bug marker %d \n", x); } while (0) ;

static const char * _cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

//
// Error checking wrapper for cutt
//
#define cuttCheck(stmt) do {                                 \
  cuttResult err = stmt;                            \
  if (err != CUTT_SUCCESS) {                                 \
  if(err == CUTT_INTERNAL_ERROR) {                           \
    printf("CUTT_INTERNAL_ERROR\n");                         \
  } else if (err == CUTT_INVALID_PARAMETER) {                \
   printf("CUTT_INVALID_PARAMETER\n");                       \
  } else if (err == CUTT_INVALID_PLAN) {                     \
   printf("CUTT_INVALID_PLAN\n");                       \
  }\
    fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
    exit(1); \
  }                                                  \
} while(0)

// Handle cuTENSOR errors
#define HANDLE_ERROR(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                   \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}


#define cublasCheck(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUBLAS_STATUS_SUCCESS )                                                   \
  { printf("Error: %s in line %d\n", _cudaGetErrorEnum(err), __LINE__); exit(-1); } \
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void
gpuAssert(cudaError_t code, const char * file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

#endif //TORQUE_GPU_LIB_HELPER_CUH
