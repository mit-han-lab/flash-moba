// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

// Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/static_switch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define EVENK_SWITCH BOOL_SWITCH

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)       \
  [&] {                                    \
    if (HEADDIM <= 64) {                   \
      constexpr static int kHeadDim = 64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kHeadDim = 256; \
      return __VA_ARGS__();                \
    }                                      \
  }()

// #define HEADDIM_SWITCH(HEADDIM, ...)       \
//   [&] {                                    \
//     constexpr static int kHeadDim = 128;   \
//     return __VA_ARGS__();                  \
//   }()

#define TOPK_SWITCH(TOPK, ...)             \
  [&] {                                    \
    if (TOPK <= 16) {                      \
      constexpr static int kTopk = 16;     \
      return __VA_ARGS__();                \
    } else if (TOPK <= 32) {               \
      constexpr static int kTopk = 32;     \
      return __VA_ARGS__();                \
    } else if (TOPK <= 64) {               \
      constexpr static int kTopk = 64;     \
      return __VA_ARGS__();                \
    }                                      \
  }()

  // #define TOPK_SWITCH(TOPK, ...)             \
  // [&] {                                    \
  //   constexpr static int kTopk = 16;     \
  //   return __VA_ARGS__();                \
  // }()
