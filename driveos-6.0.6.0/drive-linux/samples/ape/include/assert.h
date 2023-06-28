/*
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __ASSERT_H
#define __ASSERT_H

#include "compiler.h"
#include "stdio.h"

#if DEBUGLEVEL
  #define assert(x) \
    do { if (unlikely(!(x))) { \
      printf("Assert condition: %s\n%s line %d\n", #x, __FILE__, __LINE__); } \
  } while (0)
#endif

#endif

