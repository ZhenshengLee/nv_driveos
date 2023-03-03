/*
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __STDIO_H
#define __STDIO_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stddef.h"
#include "stdarg.h"
#include "printf.h"

typedef struct FILE {
	void* ctx;
} FILE;

extern FILE __stdio_FILEs[];

#define stdin  (&__stdio_FILEs[0])
#define stdout (&__stdio_FILEs[1])
#define stderr (&__stdio_FILEs[2])

FILE* fopen(const char* filename, const char* mode);
int fclose(FILE* stream);
int fseek(FILE* stream, long offset, int whence);
long ftell(FILE* stream);
int fflush(FILE* stream);

#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

size_t fwrite(const void* ptr, size_t size, size_t count, FILE* stream);
size_t fread(void* ptr, size_t size, size_t count, FILE* stream);
char* fgets(char* str, int num, FILE* fp);
int feof(FILE* stream);

int fprintf(FILE *fp, const char *fmt, ...);
int vfprintf(FILE* fp, const char* fmt, va_list ap);
int vsprintf(char* str, const char* fmt, va_list ap);

int getchar(void);

#ifdef __cplusplus
}
#endif

#endif

