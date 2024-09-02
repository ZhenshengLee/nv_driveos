/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * NvSciStream Event Loop Driven Sample App - usecase #3
 *
 * This use case consists of CUDA producer and CUDA consumer(s).
 *   It makes use of the CUDA runtime toolkit.
 *   It shows how to perform extra validation steps to meet ASIL-D safety.
 *
 * There's one data packet element, and several CRC elements, one owns by
 *   the producer and others by consumers, which contains the validaiton
 *   data.
 *
 * The producer and consumer need to get the updated CRC values by using
 *   the new *WithCrc APIs, share those CRC values and vaidate them before
 *   the critical operations.
 *
 * Before streaming, the consumer application writes all the validation
 *   data into its CRC element to share it to the producer. The producer
 *   application needs to validate the data and decide whether it's safe
 *   to proceed to streaming phase.
 *
 * During streaming phase, before the producer and consumer applications
 *   access the buffer data, they need to validate the runtime CRC value
 *   and other data to ensure the correctness.
 */

#ifndef _USECASE3_H
#define _USECASE3_H 1

/* Names for the packet elements */
#define ELEMENT_NAME_DATA           0xaa
#define ELEMENT_NAME_PROD_CRC       0xbb
#define ELEMENT_NAME_CONS_CRC_BASE  0xcc

/* Names for the endpoint info */
#define ENDINFO_NAME_STREAM_ID      0xab

/* Names for CRC data type */
#define CRC_INIT                    0xabcd
#define CRC_RUNTIME                 0xabcd

 /* Pool data structure used to store the buffer validation data */
typedef struct {
    /* Init-time valdation data magic number */
    uint32_t    magicNum;

    /* CRC value for the buffer info */
    uint32_t    buf;
} PoolInitCrcData;

/* Consumer data structure used to store all the init-time validation data */
typedef struct {
    /* Init-time valdation data magic number */
    uint32_t    magicNum;

    /*
     * Init-time CRC values
     */

    /* CRC value for producer's endpoint info */
    uint32_t    prodInfo;
    /* CRC value for the buffer info */
    uint32_t    buf;
    /* CRC value for producer's sync objects */
    uint32_t    prodSync;
    /* CRC value for consumer's sync objects */
    uint32_t    consSync;

    /*
     * Other init-time validation data
     */

    /* Stream id received from producer */
    uint32_t    streamId;
    /* Index of the consumer */
    uint32_t    consIndex;
    /* NvSciIpc endpoint name */
    char        ipcChannel[32];
} ConsInitCrcData;

/* Producer structure used to store the payload validation data */
typedef struct {
    /* Runtime valdation data magic number */
    uint32_t    magicNum;

    /*
     * Runtime CRC values
     */

     /* CRC value for producer's fences */
    uint32_t    fence;

    /*
     * Other init-time validation data
     */

     /* Stream id received from producer */
    uint32_t    frameCount;
    /* Timestamp when the payload is generated */
    uint32_t    time;
} ProdPayloadCrcData;

/* Producer structure used to store the payload validation data */
typedef struct {
    /* Runtime valdation data magic number */
    uint32_t    magicNum;

    /*
     * Runtime CRC values
     */

     /* CRC value for producer's fences */
    uint32_t    fence;
} ConsPayloadCrcData;

#endif // _USECASE3_H
