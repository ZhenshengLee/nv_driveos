/*
* Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA Corporation is strictly prohibited.
*/


FILE *adsp_fopen(const char* path, const char *modes);

int adsp_fclose(FILE *fp);

size_t adsp_fwrite(const void *data, size_t sz, size_t cnt);
size_t adsp_fread(void *data, size_t sz, size_t cnt, FILE *fp);
size_t adsp_fflush(FILE *fp);
