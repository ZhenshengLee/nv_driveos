/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMTRANSPORTINK_HPP
#define CNVMTRANSPORTINK_HPP

#include <vector>
#include <string>
#include "NvSIPLCommon.hpp"
#include "ModuleIF/CNvMCameraModule.hpp"
#include "devblk_cdi.h"
#include "utils/utils.hpp"
#include "CNvMCameraModuleCommon.hpp"

namespace nvsipl
{

/**
 * The CNvMTransportLink class encapsulates Transport Link control and configuration information.
 */
class CNvMTransportLink
{
public:
    /**
     * Values which enable the management of the programming of ser-des pairs
     */
    typedef struct {
        /**
         * Serializer device object handle
         */
        DevBlkCDIDevice* pSerCDIDevice;

        /**
         * Deserializer device object handle
         */
        DevBlkCDIDevice* pDeserCDIDevice;

        /**
         * Holding link index to be programmed by the object
         */
        std::uint8_t ulinkIndex;

        /**
         * Holds serializer native I2C address
         */
        std::uint8_t uBrdcstSerAddr;

        /**
         * Holds translation address for serializer
         */
        std::uint8_t uSerAddr;

        /**
         * All connection properties for the associated module
         */
        CNvMCameraModuleCommon::ConnectionProperty moduleConnectionProperty;

        /**
         * Flag indicating whether simulator mode has been enabled or not
         */
        bool bEnableSimulator;

        /**
         * Flag indicating whether passive mode has been enabled or not
         */
        bool bPassive;

        /**
         * Flag indicating whether the homogeneous camera support enabled or not
         */
        bool m_groupInitProg;
    } LinkParams;

    /**
     * Initialize deserializer and serializer
     * including configuration of all necessary properties for transport functionality
     * Meant to implemented for a particular module and called before a sensor has been configured
     *
     * @param[in] brdcstSerCDI      DevBlkCDIDevice serializer handle to configure
     * @param[in] linkMask          Link information used to set up config link
     * @param[in] groupInitProg     Information on whether group Init is enabled
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      NVSIPL_STATUS_NOT_SUPPORTED on invalid configuration
     * @retval                      (SIPLStatus) other propagated error
     */
    virtual SIPLStatus Init(DevBlkCDIDevice *const brdcstSerCDI, uint8_t const linkMask, bool const groupInitProg) = 0;

    /**
     * Initialization steps to be done after the sensor has been configured
     * Meant to be implemented for a particular module as needed
     * Called after a sensor has been configured
     * @param[in] brdcstSerCDI      DevBlkCDIDevice serializer handle used in configuration
     * @param[in] linkMask          Link information
     * @param[in] groupInitProg     Information on whether group Init is enabled
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
    virtual SIPLStatus PostSensorInit(DevBlkCDIDevice const* const brdcstSerCDI, uint8_t const linkMask, bool const groupInitProg) const = 0;

    /**
     * Additional initialization steps to be done after programming serializer, deserializer and sensors
     * May be called during a reconfigure
     * Meant to be implemented for a particular module as needed
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
    virtual SIPLStatus MiscInit() const = 0;

    /**
     * Start all the devices for the specific link
     * Called before sensors are started
     * Meant to be implemented for a particular module
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
    virtual SIPLStatus Start() const = 0;

    /**
     * Stop all the devices for the specific link
     * Called after sensors are stopped
     * Meant to be implemented for a particular module
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
    virtual SIPLStatus Stop() const = 0;

    /**
     * Reset all the devices for the specific link
     * Meant to be implemented for a particular module
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
    virtual SIPLStatus Reset() const = 0;

    /**
     * Deinit all the devices for the specific link
     * Meant to be implemented for a particular module
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
    virtual SIPLStatus Deinit() const = 0;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    /**
     * Get link index for the specific link
     * Meant to be implemented for a particular module
     *
     * @retval                      Link Index [0,3]
     */
    uint8_t GetLinkIndex() const {
        return m_oLinkParams.ulinkIndex;
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

    /**
     * Get whether transport link is GMSL2
     * Must be implemented for a particular module
     *
     * @retval                      true if GMSL2
     * @retval                      false if not GMSL2
     */
    virtual bool IsGMSL2() const = 0;

     /**
     * Configures the Transport Link object with the given values
     *
     * - verify params.pSerCDIDevice AND params.pDeserCDIDevice are not nullptr
     * - set m_oLinkParams = params
     *
     * @param[in] params            params used to configure the object
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      NVSIPL_STATUS_BAD_ARGUMENT on NULL ser or des handle ptr
     */
    SIPLStatus SetConfig(LinkParams const &params);

    /**
     * Default destructor
     */
    virtual ~CNvMTransportLink()= default;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if NV_IS_SAFETY
protected:
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    /**
     * Cached parameters
     */
    LinkParams m_oLinkParams;
};

} /* end of namespace nvsipl */
#endif /* CNVMTRANSPORTINK_HPP */
