/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CDI_MAX20087_H
#define CDI_MAX20087_H

#include "devblk_cdi.h"
#ifdef __cplusplus
extern "C" {
#endif
DevBlkCDIDeviceDriver*
GetMAX20087Driver(
    void);

NvMediaStatus
MAX20087Init(
    DevBlkCDIDevice const* handle);

NvMediaStatus
MAX20087CheckPresence(
    DevBlkCDIDevice const* handle);

NvMediaStatus
MAX20087SetLinkPower(
    DevBlkCDIDevice const* handle,
    uint8_t const linkIndex,
    bool const enable);

NvMediaStatus
MAX20087ReadRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
MAX20087WriteRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t const* dataBuff);

#ifdef __cplusplus
}
#endif
#endif /* CDI_MAX20087_H */
