/* NvSciStream Event Loop Driven Sample App - usecase #1
 *
 * Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto. Any use, reproduction, disclosure or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 */

/*
 * This use case consists of CUDA producer and CUDA consumer(s).
 *   It makes use of the CUDA runtime toolkit.
 *
 * There are two packet elements, a large data buffer and a small
 *   buffer containing a CRC checksum value.
 *
 * The producer operation is very simple. It takes a local source buffer
 *   filled with simple data, and issues a CUDA command to asynchronously
 *   copy it to the packet's data buffer. It generates a checksum from the
 *   source buffer and puts that in the packet's CRC buffer.
 *
 * The consumer(s) similarly issues a CUDA command to copy the packet's
 *   data buffer to a local buffer. When finished, it generates a checksum
 *   from the the local copy and compares it to the value in the packet's
 *   CRC buffer.
 *
 * The data buffer is processed through the CUDA engine, with commands issued
 *   asynchronously. Sync objects must be used to coordinate when it is safe
 *   to write and read the buffer. The CRC buffer is written and read directly
 *   through the CPU. It uses immediate mode and it is not necessary to wait
 *   for the sync objects before accessing it.
 *
 * In addition to the normal case where producers signal sync objects that
 *   consumers wait for, and vice versa, this use case also needs the
 *   producer to be able to wait for the fences it generates, in order
 *   to protect its local buffer from modification while still in use.
 *   So this use case also provides an example of CPU waiting for fences.
 */

#ifndef _USECASE1_H
#define _USECASE1_H 1

/* Names for the packet elements */
#define ELEMENT_NAME_DATA 0xdada
#define ELEMENT_NAME_CRC  0xcc

/* Names for the endpoint info */
#define ENDINFO_NAME_PROC 0xabcd

#endif // _USECASE1_H
