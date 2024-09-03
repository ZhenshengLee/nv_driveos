/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CDI_M24C32_H
#define CDI_M24C32_H

#include "devblk_cdi.h"

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
extern "C" {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif
/** @brief returns the device DEV BLK handle
 *
 * Implements:
 * - place required function methods and meta data in handle by
 *  - static DevBlkCDIDeviceDriver deviceDriverM24C32 = {
 *   - .deviceName = "M24C32",
 *   - .regLength = M24C32_REGISTER_ADDRESS_LENGTH,
 *   - .dataLength = M24C32_REGISTER_DATA_LENGTH,
 *   - .DriverCreate = DriverCreateM24C32,
 *   - .DriverDestroy = DriverDestroyM24C32,
 *   - .ReadRegister = M24C32ReadRegister,
 *   .
 *  - }
 *  .
 * @return pointer to deviceDriverM24C32  */
DevBlkCDIDeviceDriver*
GetM24C32Driver(
    void);

/** @brief checks the presence of the M24C32 device
 *
 * Impements:
 * - extracts device driver handle from handle
 * - verify device driver handle is not NULL
 * - Attempt to read the first byte on the EEPROM chip by
 *  - status = DevBlkCDII2CPgmrReadUint8(
 *   - drvHandle->i2cProgrammer,
 *   - (uint16_t)M24C32_FIRST_ADDRESS,
 *   - &data)
 *   .
 *  .
 * @param[in] handle DEV BLK handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code
 * when any of the called functions returns error */
NvMediaStatus
M24C32CheckPresence(
   DevBlkCDIDevice const* handle);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif
#endif /* CDI_M24C32_H */
