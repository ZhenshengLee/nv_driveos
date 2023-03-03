/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CDI_N24C64_H_
#define _CDI_N24C64_H_

#include "devblk_cdi.h"

DevBlkCDIDeviceDriver*
GetN24C64Driver(
    void);

NvMediaStatus
N24C64CheckPresence(
    DevBlkCDIDevice *handle);

NvMediaStatus
N24C64ReadRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
N24C64WriteRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t const* dataBuff);

#endif /* _CDI_N24C64_H_ */
