/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/*
 * NvSciStream Event Loop Driven Sample App - usecase #2
 *
 * This use case consists of NvMedia producer and CUDA consumer(s).
 *   It makes use of the CUDA runtime toolkit.
 *
 * There is a single packet element containing an image. The producer
 *   performs a blit from a local source buffer to the packet. The consumer
 *   performs a copy from the packet to a local target buffer. Unlike
 *   use case 1, there is no checksum to validate the operations.
 */

#ifndef _USECASE2_H
#define _USECASE2_H 1

/* Names for the packet elements */
#define ELEMENT_NAME_IMAGE 0xbeef

/* Names for the endpoint info */
#define ENDINFO_NAME_PROC 0xabcd

#endif // _USECASE2_H
