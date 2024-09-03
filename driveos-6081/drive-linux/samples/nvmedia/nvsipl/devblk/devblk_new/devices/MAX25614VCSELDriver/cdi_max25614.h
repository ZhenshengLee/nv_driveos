/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CDI_MAX25614_H
#define CDI_MAX25614_H

#include "devblk_cdi.h"
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
extern "C" {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

typedef struct {
    uint8_t iBoost;
    uint8_t vBoost;
    uint8_t iLed;
    uint8_t tOnMax;
    uint8_t tSlew;
    uint8_t vLedMax;
} MAX25614Params;

DevBlkCDIDeviceDriver*
GetMAX25614Driver(
    void);

NvMediaStatus
MAX25614Init(
    DevBlkCDIDevice const* handle,
    MAX25614Params params);

NvMediaStatus
MAX25614CheckPresence(
    DevBlkCDIDevice const* handle);

NvMediaStatus
MAX25614ReadRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
MAX25614WriteRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t const* dataBuff);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif
#endif /* CDI_MAX25614_H */
