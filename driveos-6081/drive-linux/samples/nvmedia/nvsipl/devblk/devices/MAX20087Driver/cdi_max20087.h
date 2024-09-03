/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
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
    DevBlkCDIRootDevice* const cdiRootDev,
    DevBlkCDIDevice const* handle);

NvMediaStatus
MAX20087CheckPresence(
    DevBlkCDIDevice const* handle);

/**
 * Use power switch to power on/off the camera modules.
 *
 * @param[in] handle                Handle to the power device
 * @param[in]  linkIndex            The data link index;
 *                                  Valid range: [0, (Maximum Links Supported per Deserializer - 1)]
 * @param[in]  enable               True to turn on power, False to turn off power
 * @param[in] interruptMaskState    (Used in FUSA driver)
 *                                  Interrupt mask state
 *                                  (INTERRUPT_MASK_INITED_STATE = 0) - Interrupts not masked
 *                                  (INTERRUPT_GLOBAL_MASKED_STATE = 1) - Interrupts masked
                                    (INTERRUPT_MASK_RESTORED_STATE = 2) - Interrupts restored
 * @param[out] savedInterruptMask   (Used in FUSA driver)
                                    Stores updates of mask register (enable/disable bits) when
                                    interrupts are in masked state.
                                    Used when restoring mask register.
 *
 * @retval      NVMEDIA_STATUS_OK On Success
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER when incorrect parameters are passed.
 * @retval      NVMEDIA_STATUS_ERROR error status propagated
 */
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
