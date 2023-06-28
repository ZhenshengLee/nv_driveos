/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __LIB_APRALLOC_H
#define __LIB_APRALLOC_H

void dramapr_init(void);
void *dramapr_alloc(size_t, unsigned int alignment);
void dramapr_free(void *);

void aramapr_init(void);
void *aramapr_alloc(size_t, unsigned int alignment);
void aramapr_free(void *);

#endif /* __LIB_APRALLOC_H */
