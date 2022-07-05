
#ifndef TORQUE_GPU_LIB_HELPER_CUH
#define TORQUE_GPU_LIB_HELPER_CUH

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

#endif //TORQUE_GPU_LIB_HELPER_CUH
