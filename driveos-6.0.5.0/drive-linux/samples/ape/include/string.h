/*
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __LIB_STRING_H
#define __LIB_STRING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* memchr(void const*, int, size_t);
int memcmp(void const*, const void *, size_t);
void* memcpy(void*, void const*, size_t);
void* memmove(void*, void const*, size_t);
void* memset(void*, int, size_t);

char* strchr(char const*, int);
int strcmp(char const*, char const*);
char* strcpy(char*, char const*);
size_t strlen(char const*);
char* strstr(char const*, char const*);

int strncmp(char const*, char const*, size_t);
char* strncpy(char*, char const*, size_t);

#ifdef __cplusplus
}
#endif

#endif
