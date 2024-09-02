/* NvSciStream Safety Sample App - Usecase
 *
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/* Application information:
* This sample app demonstartes the communication of
* ASIL process with QM processes where one QM process(say QM proxy) resides
* on the same SOC where ASIL process resides and another
* QM process(say QM2) resides on different SOC.
* The QM process(QM proxy) acts as a proxy process which does not share the
* ASIL process buffers with the other QM process(QM2). Instead, the
* data copy happens beteen the buffers from ASIL process to QM2
* process
* ASIL process pipeline: Producer->Pool->Multicast->Queue->Consumer1
*                                       |->Limiter -> ReturnSync->IpcSrc
* QM process(QM proxy) pipeline: IpcDst->Queue->ProxyApp->Pool->Queue->C2Csrc
* QM process(QM2) pipeline: C2CDst->Pool->Queue->Consumer2
* ProxyApp -> Introduces a Consumer and Producer where consumer retrieves the
* data from ASIL buffers and writes to Producer buffers to send the data to
* QM2 process.
*/

#ifndef _USECASE_H
#define _USECASE_H 1

/* Names for the packet elements */
#define ELEMENT_NAME_DATA 0xdada
#define ELEMENT_NAME_CRC  0xcc

/* Names for the endpoint info */
#define ENDINFO_NAME_PROC 0xabcd

#endif // _USECASE_H
