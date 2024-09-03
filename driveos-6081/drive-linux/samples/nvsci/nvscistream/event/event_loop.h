/* NvSciStream Event Loop Driven Sample App - event handler abstraction
 *
 * Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _EVENT_LOOP_H
#define _EVENT_LOOP_H 1

#include <stdint.h>
#include "nvscistream.h"

/*
 * Block event handler function.
 *   Input:
 *     data: The block's type-specific private data structure
 *     wait: Flag indicating whether to wait for an event
 *   Returns:
 *     +2: Block has completed its last operation and will be destroyed
 *     +1: An event was found and processed
 *      0: No event was found (not an error)
 *     -1: Block has encountered a fatal error and will be destroyed
 */
typedef int32_t (*BlockFunc)(void* data, uint32_t wait);

/* Table of events to abstract the two approaches for event loops */
typedef struct {
    int32_t (*init)(void);
    int32_t (*reg)(NvSciStreamBlock, void*, BlockFunc);
    int32_t (*regInt)(NvSciStreamBlock);
    int32_t (*loop)(void);
} EventFuncs;

/* Chosen event function table */
extern EventFuncs const* eventFuncs;

/* Event tables for the two methods */
extern EventFuncs const eventFuncs_Service;
extern EventFuncs const eventFuncs_Threads;

#endif // _EVENT_LOOP_H
