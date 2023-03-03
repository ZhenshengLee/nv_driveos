/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMCAMPWR_HPP
#define CNVMCAMPWR_HPP

#include "NvSIPLPlatformCfg.hpp"
extern "C" {
#include "devblk_cdi.h"
}
#include "CNvMDevice.hpp"
#include "utils/utils.hpp"

namespace nvsipl
{

/**
 * The CNvMCampwr class encapsulates camera power device control and configuration information.
 */
class CNvMCampwr : public CNvMDevice
{
public:
    /**
    * Default Destructor
    */
    virtual ~CNvMCampwr() = default;

    /**
     * Sets the configuration for the camera power device object based on the params
     * Updates the object's state information to indicate that configuration has been set
     *
     * If simulator mode is not enabled and passive mode is not enabled,
     * this will save thecamera power device's I2C address and register it with the address manager.
     *
     * @param[in] i2cAddress        camera power device I2C address
     * @param[in] params            Device information used to register I2C address
     * @retval                      NVSIPL_STATUS_OK on completion
     */
    virtual SIPLStatus SetConfig(uint8_t i2cAddress, const DeviceParams *const params);

    /**
     * Sets the object's driver handle to the provided driver
     *
     * @param[in] driver            DevBlkCDIDeviceDriver ptr to store
     */
    void SetDriverHandle(DevBlkCDIDeviceDriver * const driver) {m_pCDIDriver = driver;}

     /**
     * Gets camera power load switch error size
     *
     * Gets size of camera power load switch errors to be used by the client for allocating buffers.
     *
     * @param[out] errorSize    size_t size of camera power load switch error information
     *                          (0 if no valid size found).
     *
     * @retval      NVSIPL_STATUS_OK on successful completion
     * @retval      NVSIPL_STATUS_NOT_SUPPORTED if not implemented for a particular driver
     * @retval      (SIPLStatus) error status propagated
     */
    virtual SIPLStatus GetErrorSize(size_t & errorSize);

    /**
     * Gets generic camera power load switch error information
     *
     * Gets detailed camera power load switch error information and populates a provided buffer.
     * This is expected to be called after the client is notified of errors.
     *
     * @param[out] buffer       Buffer to populate with error information
     * @param[in]  bufferSize   Size of buffer to read to
     * @param[out] size         Size of data read to the buffer
     *
     * @retval      NVSIPL_STATUS_OK on successful completion
     * @retval      NVSIPL_STATUS_NOT_SUPPORTED if not implemented for a particular driver
     * @retval      (SIPLStatus) error status propagated
     */
    virtual SIPLStatus GetErrorInfo(std::uint8_t * const buffer,
                                    const std::size_t bufferSize,
                                    std::size_t &size);

    /**
     * Check to see if a power control device is supported
     *
     * @retval      NVSIPL_STATUS_OK if a power control device is supported
     * @retval      NVSIPL_STATUS_NOT_SUPPORTED if power control is not supported
     */
    virtual SIPLStatus isSupported();

    /**
     * Use power control device to power on/off the camera modules.
     *
     * @param[in]  cdiDev      Handle to the power device
     * @param[in]  linkIndex   The data link index;
     *                         Valid range: [0, (Maximum Links Supported per Deserializer - 1)]
     * @param[in]  enable      True to turn on power, False to turn off power
     *
     * @retval      NVSIPL_STATUS_OK on successful completion
     * @retval      NVSIPL_STATUS_ERROR if control failed
     */
    virtual SIPLStatus PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev, uint8_t const linkIndex, bool const enable);

    /**
     * Creates a new power control CDI device
     *
     * @param[in]  cdiRootDev  The CDI root device handle
     * @param[in]  linkIndex   The data link index;
     *                         Valid range: [0, (Maximum Links Supported per Deserializer - 1)]
     *
     * @retval      A CDI device handle or NULL if error occurred
     */
    virtual SIPLStatus CreatePowerDevice(DevBlkCDIRootDevice* const cdiRootDev, const uint8_t linkIndex);

    /**
     * Retrieves the CDI device handle for the power device
     *
     * @retval      A CDI device handle or NULL if error occurred
     */
    virtual DevBlkCDIDevice* GetDeviceHandle();

    /**
     * Initializes the power device object
     *
     * @param[in]  cdiDev      The CDI device handle
     * @param[in]  linkIndex   The data link index;
     *                         Valid range: [0, (Maximum Links Supported per Deserializer - 1)]
     *
     * @retval      NVSIPL_STATUS_OK Power device initialized succefully
     * @retval      NVSIPL_STATUS_ERROR Power device failed to initialize
     */
    virtual SIPLStatus InitPowerDevice(DevBlkCDIDevice* const cdiDev, uint8_t const linkIndex);
};

} // end of namespace nvsipl
#endif //CNVMCAMPWR_HPP