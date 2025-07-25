// clang-format off
#ifndef TT_KERNEL_{func_name[1]}_INCLUDES
#define TT_KERNEL_{func_name[1]}_INCLUDES

#include "cnrt.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

// tt-linker: {func_name[0]}:{signature}:{tt_jit_name}
/*
{func_docstring}
*/
cnrtRet_t {func_name[0]}(cnrtQueue_t queue, cnrtDim3_t* dim, {signature});

#ifdef __cplusplus
}}
#endif

#endif
