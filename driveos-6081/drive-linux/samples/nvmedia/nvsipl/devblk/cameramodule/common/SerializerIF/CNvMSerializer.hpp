/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMSERIALIZER_HPP
#define CNVMSERIALIZER_HPP

#include "NvSIPLPlatformCfg.hpp"
#include "devblk_cdi.h"

#include "CNvMDevice.hpp"
#include "utils/utils.hpp"
#include "INvSIPLDeviceInterfaceProvider.hpp"

namespace nvsipl
{

/**
 * The CNvMSerialzer class encapsulates serializer control and configuration information.
 */
class CNvMSerializer : public CNvMDevice
{
public:
    //! Destructor
    virtual ~CNvMSerializer() = default;

    /**
     * Set the Serializer object's device parameters.
     * If simulator mode is not enabled and passive mode is not enabled,
     * this will save the serializer's I2C address and register it with the address manager.
     *
     * @param[in] serializerInfo    Serializer info struct containing I2C address
     * @param[in] params            Device information used to register I2C address
     * @retval                      NVSIPL_STATUS_OK on completion
     */
    virtual SIPLStatus SetConfig(SerInfo const *const serializerInfo, DeviceParams *const params) = 0;

     /**
     * Gets serializer error size
     *
     * Gets size of serializer errors to be used by the client for allocating buffers.
     *
     * @param[out] errorSize    size_t size of serializer error information
     *                          (0 if no valid size found).
     *
     * @retval      NVSIPL_STATUS_OK on successful completion
     * @retval      NVSIPL_STATUS_NOT_SUPPORTED if not implemented for a particular driver
     * @retval      (SIPLStatus) error status propagated
     */
    virtual SIPLStatus GetErrorSize(size_t & errorSize) const = 0;

    /**
     * Gets generic serializer error information
     *
     * Gets detailed serializer error information and populates a provided buffer.
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
    virtual SIPLStatus GetErrorInfo(std::uint8_t * const buffer, std::size_t const bufferSize, std::size_t &size) const = 0;

    /**
     * @brief Get CDI Handle
     *
     * @retval pointer to DevBlkCDIDevice
     */
    inline DevBlkCDIDevice* GetSerializerDeviceHandle(void) const {
        return CNvMDevice::GetCDIDeviceHandle();
    };
};

} /* end of namespace nvsipl */
#endif /* CNVMSERIALIZER_HPP */

