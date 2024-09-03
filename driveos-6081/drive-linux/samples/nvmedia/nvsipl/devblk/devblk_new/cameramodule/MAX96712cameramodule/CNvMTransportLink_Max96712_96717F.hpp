/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef CNVMTRANSPORTINK_MAX96712_96717F_HPP
#define CNVMTRANSPORTINK_MAX96712_96717F_HPP
#include "TransportLinkIF/CNvMTransportLink.hpp"
namespace nvsipl
{
class CNvMTransportLink_Max96712_96717F: public CNvMTransportLink
{
public:
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
    virtual SIPLStatus Init(DevBlkCDIDevice *const brdcstSerCDI, uint8_t const linkMask, bool const groupInitProg);

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
    SIPLStatus PostSensorInit(uint8_t const linkMask, bool const groupInitProg) const override;

    /**
     * Additional initialization steps to be done after programming serializer, deserializer and sensors
     * May be called during a reconfigure
     * Meant to be implemented for a particular module as needed
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
     SIPLStatus MiscInit() const override;

    /**
     * Start all the devices for the specific link
     * Called before sensors are started
     * Meant to be implemented for a particular module
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
      SIPLStatus Start() const override
    {
        return NVSIPL_STATUS_OK;
    };

    /**
     * Deinit all the devices for the specific link
     * Meant to be implemented for a particular module
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
     SIPLStatus Deinit() const override
    {
        return NVSIPL_STATUS_OK;
    };

    /**
     * Stop all the devices for the specific link
     * Called after sensors are stopped
     * Meant to be implemented for a particular module
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
    SIPLStatus Stop() const override
    {
        return NVSIPL_STATUS_OK;
    };

    /**
     * Reset all the devices for the specific link
     * Meant to be implemented for a particular module
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      (SIPLStatus) other implementation-determined or propagated error
     */
    SIPLStatus Reset() const override
    {
        return NVSIPL_STATUS_OK;
    };

    /**
     * Get whether transport link is GMSL2
     * Must be implemented for a particular module
     *
     * @retval                      true if GMSL2
     * @retval                      false if not GMSL2
     */
     bool IsGMSL2() const override
    {
        return true;
    };

private:

    bool initDone = false;

    /* Flag to indicate if MAX96717(F) Serializer Register CRC (SM27) is enabled */
    mutable bool m_bSerRegCrcEnabled = false;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    // Dump link Parameters
    void DumpLinkParams() const;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    // Setup address translations
    SIPLStatus SetupAddressTranslations(DevBlkCDIDevice const* const brdcstSerCDI) const;

    /**
     * @brief Check DES Revision, input format, Link status and setup SER i2c
     *
     * @param[in] brdcstSerCDI  Broadcast serializer Device Driver handle
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval NVSIPL_STATUS_BAD_PARAMETER on invalid parameter
     * @retval NVSIPL_STATUS_ERROR in failure case
     */
    SIPLStatus BasicCheckModule(DevBlkCDIDevice const* const brdcstSerCDI) const;

    /**
     * Enable/Disable EEPROM Write Protection by setting EEPROM_WP pin (MFP7)
     * output in Serializer.
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus SetEEPROMWriteProtect(void) const;

    /**
     * @brief Serializer Clock Configuration
     *
     * @retval NVSIPL_STATUS_OK on successful completion.
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus SetSERClk(void) const;

    /**
     * @brief gpio connection configure between Serializer and De-Serializer
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus SetSERDESGpioForward(void) const;

    /**
     * @brief Serializer PHY configuration
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus SetSERPhy(void) const;

    /**
     * @brief gpio connection configure between Serializer and De-Serializer
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus SetSERVideo(void) const;
    /**
     * @brief Serializer FRSYNC GPIO level configuration
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus SetSERFsyncLevel(void) const;
    /**
     * @brief Serializer GPIO configuration
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus SetSERFsync(void) const;

    /**
     * @brief Release sensor reset and sets SMs reporting
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus ReleaseResetAndErrReporting(void) const;

    /**
     * @brief Init Errb related settings based on sensor property
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus Init_Errb(const SIPLStatus instatus) const;
#if FUSA_CDD_NV
    /**
     * @brief VPRBSDiagnosticTest
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus VPRBSDiagnosticTest(void) const;

    /**
     * @brief ERRG Diagnostic test
     *
     * @retval NVSIPL_STATUS_OK on successful completion
     * @retval (SIPLStatus)     other error propagated
     */
    SIPLStatus ERRGDiagnosticTest(void) const;
#endif
};
} // end of namespace nvsipl
#endif /* CNVMTRANSPORTINK_MAX96712_96717F_HPP */
