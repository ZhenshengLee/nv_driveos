/*
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __STDLIB_H
#define __STDLIB_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int abs(int n);
long int labs (long int n);
int atoi(const char* str);
long atol(const char* str);
float atof(const char* str);

#ifdef __cplusplus
}
#endif

#endif

