/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMMAX96712_FUSA_NV_HPP
#define CNVMMAX96712_FUSA_NV_HPP

#include "DeserializerIF/CNvMDeserializer.hpp"
#include "NvSIPLDeviceBlockTrace.hpp"
#include "MAX96712_Fusa_nv_CustomInterface.hpp"
#include "utils/utils.hpp"
#include "cdi_max96712_nv.h"
#include "cdd_nv_error.h"

namespace nvsipl
{

/** defines the buffer size for storing all desrializer error set */
const uint64_t MAX_DESER_ERROR_SIZE = ((sizeof(std::uint8_t) + sizeof(GlobalFailureTypeMAX96712))*
                                       MAX96712_MAX_GLOBAL_ERROR_NUM) +
                                      ((sizeof(std::uint8_t) + sizeof(PipelineFailureTypeMAX96712))*
                                       MAX96712_NUM_VIDEO_PIPELINES*MAX96712_MAX_PIPELINE_ERROR_NUM) +
                                      ((sizeof(std::uint8_t) + sizeof(LinkFailureTypeMAX96712))*
                                       MAX96712_MAX_NUM_LINK*MAX96712_MAX_LINK_BASED_ERROR_NUM);

/** Class for Max96712 deserialzer */
class CNvMMax96712_Fusa : public CNvMDeserializer, public MAX96712FusaNvCustomInterface
{
public:
    CNvMMax96712_Fusa();

    virtual ~CNvMMax96712_Fusa();

    /**
     * @brief Enables links
     * - Get the links currently enabled by:
     *  - nvmStatus = MAX96712ReadParameters(
     *   - m_upCDIDevice.get(),
     *   - CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS,
     *   - sizeof(int32_t),
     *   - &readParamsMAX96712)
     *   .
     *  .
     * - if failed - error exit ConvertNvMediaStatus(nvmStatus)
     * - if readParamsMAX96712.link != linkMask
     *  - Enabling links by
     *   - preparing a parameter
     *    - WriteParametersParamMAX96712 writeParamsMAX96712 = {};
     *    - writeParamsMAX96712.link = (static_cast<LinkMAX96712>(linkMask));
     *    .
     *   - write parameter to deserializer by:
     *    - nvmStatus = MAX96712WriteParameters(
     *     - m_upCDIDevice.get(),
     *     - CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS,
     *     - sizeof(int32_t),
     *     - &writeParamsMAX96712)
     *     .
     *    -
     *   - if failed - error exit ConvertNvMediaStatus(nvmStatus)
     *   - sleep 15 msec by
     *    - std::this_thread::sleep_for<>(std::chrono::milliseconds(15));
     *    .
     *   .
     * @param[in] linkMask  link masks to be enabled
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    virtual SIPLStatus EnableLinks(uint8_t const linkMask);

    /** @brief Control Links
     * - Get the links currently enabled into readParamsMAX96712 by:
     *  - nvmStatus = MAX96712ReadParameters(
     *   - m_upCDIDevice.get(),
     *   - CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS,
     *   - sizeof(int32_t),
     *   - &readParamsMAX96712)
     *   .
     *  .
     * - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     * - for each item in linkActions
     *  - Note: Bit 7 is the flag of power cycling.
     *   - Without power reset, remote ERRB needs to be re-enabled after link locked
     *  - if ((item.linkIdx & 0x80U) == 0x80U)
     *   - set powerReset = 1U
     *   .
     *  - set linkIdx = item.linkIdx & 0x7FU
     *  - if linkIdx >= MAX96712_MAX_NUM_LINK , i.e. invalid
     *   - error exit NVSIPL_STATUS_ERROR
     *   .
     *  - set linkBit = (static_cast<uint8_t>(1U) << linkIdx)
     *  - if item.eAction == LinkAction::LINK_ENABLE
     *   - set link |= linkBit;
     *   - set linkMask |= linkBit;
     *   - set link_enabled = true;
     *   .
     *  - elif item.eAction == LinkAction::LINK_DISABLE
     *   - set link &= (static_cast<uint8_t>(~linkBit) & 0x0FU)
     *   - prepare a parameter in writeParamsMAX96712 and apply it:
     *    - writeParamsMAX96712.linkIndex = linkIdx
     *    - nvmStatus = MAX96712WriteParameters(
     *     - m_upCDIDevice.get(),
     *     - CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK,
     *     - sizeof(uint8_t),
     *     - &writeParamsMAX96712);
     *     .
     *    .
     *   - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     *  - elif item.eAction == LinkAction::LINK_NO_ACTION
     *   - do nothing
     *   .
     *  - else
     *   - error exit NVSIPL_STATUS_BAD_ARGUMENT
     *   .
     *  .
     * - if (linkActions.size() > 0U)
     *  - prepare a parameter in writeParamsMAX96712 and apply it:
     *   - writeParamsMAX96712.link = static_cast<LinkMAX96712>(link)
     *   - nvmStatus = MAX96712WriteParameters(
     *     - m_upCDIDevice.get(),
     *     - CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS,
     *     - sizeof(int32_t),
     *     - &writeParamsMAX96712)
     *     .
     *   - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     *   - check if links are locked by calling CheckLinkLock(linkMask)
     *   - if links are locked and (powerReset == 0U)
     *    - Restore ERRB Rx Eanble and Video Pipeline Enable bit, by:
     *     - for each item in linkActions
     *      - if (item.eAction == LinkAction::LINK_ENABLE)
     *       - prepare a parameter in writeParamsMAX96712 and apply it:
     *        - writeParamsMAX96712.linkIndex = (item.linkIdx & 0x7FU);
     *        - nvmStatus = MAX96712WriteParameters(
     *         - m_upCDIDevice.get(),
     *         - CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK,
     *         - sizeof(uint8_t),
     *         - &writeParamsMAX96712);
     *         .
     *        - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     *        .
     *       .
     *      .
     *     .
     *    .
     *   .
     *  .
     * @param[in] linkActions a list of link actions
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    virtual SIPLStatus ControlLinks(const std::vector<LinkAction>& linkActions) override;

    /**
     * @brief Check Link Lock
     * - for each item in m_ovLinkModes
     *  - if (item.linkIndex < MAX96712_MAX_NUM_LINK) ,i.e. is valid
     *   - if linkMask does not contains bit for item.linkIndex, (1 << item.linkIndex)
     *    - do nothing for this item (continue)
     *   - if item.elinkMode is either LinkMode::LINK_MODE_GMSL2_6GBPS or LinkMode::LINK_MODE_GMSL2_3GBPS
     *    - Check config link lock by:
     *     - nvmStatus = MAX96712CheckLink(
     *      - m_upCDIDevice.get(),
     *      - GetMAX96712Link(item.linkIndex),
     *      - CDI_MAX96712_LINK_LOCK_GMSL2,
     *      - false)
     *      .
     *     .
     *    - if (nvmStatus == NVMEDIA_STATUS_TIMED_OUT)
     *     - perform oneshot reset by
     *      -  MAX96712OneShotReset(
     *       - NVMEDIA_STATUS_OK,
     *       - m_upCDIDevice.get(),
     *       - GetMAX96712Link(item.linkIndex))
     *       .
     *      .
     *     - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     *     - Check config link lock by:
     *      - nvmStatus = MAX96712CheckLink(
     *       - m_upCDIDevice.get(),
     *       - GetMAX96712Link(item.linkIndex),
     *       - CDI_MAX96712_LINK_LOCK_GMSL2,
     *       - false)
     *       .
     *      .
     *     - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     *     .
     *    - else
     *     - exit error ConvertNvMediaStatus(nvmStatus)
     *     .
     *   -
     *  - else
     *   -  error exit NVSIPL_STATUS_NOT_SUPPORTED
     *   .
     * .
     * @param[in] linkMask link mask of links to be checked
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    virtual SIPLStatus CheckLinkLock(uint8_t const linkMask) override;

    /** @brief returns the number of errors supported by driver
     * @param[out] errorSize pointer for storing max error code
     * @returns NVSIPL_STATUS_OK
     */
    virtual SIPLStatus GetErrorSize(size_t & errorSize) override;

    /**
     * @brief Gets detailed error information and populates a provided buffer.
     *
     * This is expected to be called after the client is notified of errors.
     *
     * If no error info is expected (max error size is 0), this can be called with
     * null buffer to retrieve only the remote and link error information.
     *
     * - sleep for 30 msec
     * - get error status from deserializer, by
     *  - nvmStatus = MAX96712GetErrorStatus(
     *   - m_upCDIDevice.get(),
     *   - static_cast<uint32_t>(sizeof(errorStatus)),
     *   -  &errorStatus)
     *   .
     *  .
     * - if failed - error exit ConvertNvMediaStatus(nvmStatus)
     * - Check whether serializer error is detected as well, by:
     *  - nvmStatus = MAX96712GetSerializerErrorStatus(
     *  - m_upCDIDevice.get(),
     *   - &isRemoteError)
     *   .
     *  .
     * - verify ((buffer != nullptr) && (errorStatus.count != 0U))
     * - calculate sub-errors buffer sizes:
     *  - sizeMax96712ErrorStatusGlobal   = (sizeof(errorStatus.globalRegVal))   + (sizeof(errorStatus.globalFailureType))
     *  - sizeMax96712ErrorStatusPipeline = (sizeof(errorStatus.pipelineRegVal)) + (sizeof(errorStatus.pipelineFailureType))
     *  - sizeMax96712ErrorStatusLink     = (sizeof(errorStatus.linkRegVal))     + (sizeof(errorStatus.linkFailureType))
     *  .
     * - calculate total buffer sizes
     *  -  sizeMax96712ErrorStatus = sizeMax96712ErrorStatusGlobal + sizeMax96712ErrorStatusPipeline + sizeMax96712ErrorStatusLink
     *  .
     * - verify sizeMax96712ErrorStatus == MAX_DESER_ERROR_SIZE
     *  - if not, error exit NVSIPL_STATUS_ERROR
     *  .
     * - verify user provided bufferSize >= MAX_DESER_ERROR_SIZE
     *  - if not error exit NVSIPL_STATUS_BAD_ARGUMENT
     * . copy all error information into user buffer
     *  - std::size_t sizeIncr = 0U;
     *  - fusa_memcpy(&tmpBuf[sizeIncr], &errorStatus.globalFailureType, sizeof(errorStatus.globalFailureType));
     *  - sizeIncr += sizeof(errorStatus.globalFailureType);
     *  - fusa_memcpy(&tmpBuf[sizeIncr], &errorStatus.globalRegVal, sizeof(errorStatus.globalRegVal));
     *  - sizeIncr += sizeof(errorStatus.globalRegVal);
     *  - fusa_memcpy(&tmpBuf[sizeIncr], &(errorStatus.pipelineFailureType), sizeof(errorStatus.pipelineFailureType));
     *  - sizeIncr += sizeof(errorStatus.pipelineFailureType);
     *  - fusa_memcpy(&tmpBuf[sizeIncr], &(errorStatus.pipelineRegVal), sizeof(errorStatus.pipelineRegVal));
     *  - sizeIncr += sizeof(errorStatus.pipelineRegVal);
     *  - fusa_memcpy(&tmpBuf[sizeIncr], &(errorStatus.linkFailureType), sizeof(errorStatus.linkFailureType));
     *  - sizeIncr += sizeof(errorStatus.linkFailureType);
     *  - fusa_memcpy(&tmpBuf[sizeIncr], &(errorStatus.linkRegVal), sizeof(errorStatus.linkRegVal));
     *  - size = sizeMax96712ErrorStatus;
     *  - fusa_memcpy(buffer, static_cast<void *>(&tmpBuf[0]), sizeof(tmpBuf));
     *
     *
     *
     * @param[out] buffer           A byte pointer Buffer (of type  uint8_t) to populate
     *                              with error information. It cannot be NULL.
     * @param[in]  bufferSize       Size (of type  size_t) of buffer to read to. Should be in
     *                              range of any value from 0 to the maximum size of an
     *                              allocation (0 if no valid size found). Error buffer size is
     *                              device driver implementation specific.
     * @param[out] size             Size (of type  size_t) of data read to the buffer
     *                              (0 if no valid size found).
     * @param[out] isRemoteError    A flag (of type bool) set to true if remote serializer
     *                              error detected.
     * @param[out] linkErrorMask    Mask (of type  uint8_t) for link error state
     *                              (1 in index position indicates error).
     *                              Expected range is [0x0U, 0xFU].
     *
     * @retval      NVSIPL_STATUS_OK            On successful completion.
     * @retval      NVSIPL_STATUS_NOT_SUPPORTED Not supported/implemented by this
     *                                          Deserializer instance.
     * @retval      (SIPLStatus)                Error status propagated (Failure).
     */
    virtual SIPLStatus GetErrorInfo(std::uint8_t * const buffer, const std::size_t bufferSize,
                            std::size_t &size, bool & isRemoteError,
                            std::uint8_t& linkErrorMask) override;

    /**
     * @brief Device specific implementation to power control Deserializer
     *
     * When @a powerOn is true,  prepares the deserializer for I2C
     * communication (enable power gates, request ownership from a board
     * management controller, etc.). However, the deserializer will not be
     * initialized when this is invoked, so implementations should not assume
     * Init() has been performed.
     *
     * When @a powerOn is false, the inverse of the operations for powerup
     * should be performed. This will be called after Deinit(), and after
     * deinit and poweroff for all device connected to this deserializer.
     *
     * - get deserializer power control methid by:
     *  - nvmStatus = DevBlkCDIGetDesPowerControlMethod(
     *   - m_upCDIDevice.get(),
     *  - &m_pwrMethod)
     *  .
     * - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     * - if m_pwrMethod == 0
     *  - note: Default is NvCCP, other power backends can be used here based on platform/usecase
     *  - status = PowerControlSetAggregatorPower(
     *   . m_pwrPort,
     *   - m_oDeviceParams.bPassive,
     *   - powerOn)
     *   .
     *  .
     *
     * @param[in] powerOn   A flag, when true indicates the deserializer should
     *                      be powered on, and when false indicates that the
     *                      deserializer should be powered off.
     *
     * @retval NVSIPL_STATUS_OK When link power has been modified successfully
     * @retval (SIPLStatus)     Error propagated (Failure).
     *
     */
    virtual SIPLStatus DoSetPower(bool const powerOn) override final;

    /**
     * @brief Get interface provided by this driver matching a provided ID
     *
     * This API is used to get access to device specific custom interfaces (if
     * any) defined by this Deserializer, which can be invoked directly by
     * Device Block clients.
     *
     * The interface is not supported
     *
     * @param[in] interfaceId   Unique identifier (of type UUID) for the
     *                          interface to retrieve.
     *
     * @retval    nullptr - not implemented.
     *
     */
    virtual Interface* GetInterface(const UUID &interfaceId) override;

    virtual SIPLStatus GetInterruptStatus(
        const uint32_t gpioIdx,
        IInterruptNotify &intrNotifier) override;

    /**
     * @brief Get deserializer overflow error information. Checks for line
     *        memory overflow and cmd buffer overflow.
     *
     * @param[in] customErrInfo Structure to hold the error information.
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     * @retval NVSIPL_STATUS_ERROR Any system error.
     */
    virtual SIPLStatus GetOverflowError(
        DeserializerOverflowErrInfo *const customErrInfo
    ) const override;

    /**
     * @brief Get deserializer CSI PLL Lock status.
     *
     * @param[in] customErrInfo Structure to hold the error information.
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     * @retval NVSIPL_STATUS_ERROR Any system error.
     */
    virtual SIPLStatus GetCSIPLLLockStatus(
        DeserializerCSIPLLLockInfo *const customErrInfo
    ) const override;

    /**
     * @brief Get deserializer RT flags status.
     *
     * @param[in] customErrInfo Structure to hold the error information.
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     * @retval NVSIPL_STATUS_ERROR Any system error.
     */
    virtual SIPLStatus GetRTFlagsStatus(
        DeserializerRTFlagsInfo *const customErrInfo
    ) const override;

    /**
     * @brief Get Deserializer VID_SEQ_ERR bits status.
     *
     * @param[in] customErrInfo Structure to hold the error information.
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     * @retval NVSIPL_STATUS_ERROR Any system error.
     */
    SIPLStatus GetVidSeqErrStatus(
        DeserializerVidSeqErrInfo *const customErrInfo
    ) const override;

    /**
     * @brief Set the heterogeneous frame sync
     *
     * @param[in] muxSel Multiplexer selection
     * @param[in] gpioNum GPIO index to be used to supply the frame sync per camera
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     * @retval NVSIPL_STATUS_ERROR Any system error.
     */
    virtual SIPLStatus SetHeteroFrameSync(uint8_t const muxSel,
                                          uint32_t const gpioNum[MAX_CAMERAMODULES_PER_BLOCK]) const override;

    /**
     * @brief Get the camera group ID
     *
     * @param[in] grpId camera group ID
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     */
    virtual SIPLStatus GetGroupID(uint32_t &grpId) const override;
protected:
    /**
     * @brief Set Max96712 context
     * - allocate new device driver context structure in driverContext
     * - reset values in context memory - call m_upDrvContext.reset(driverContext)
     * - get context pointer from  ctx = &driverContext->m_Context
     * - call SetCtxInitValue_CNvMMax96712Fusa
     * - call UpdateTxPort_CNvMMax96712Fusa
     * - call UpdateMipiSpeed_CNvMMax96712Fusa
     * - Set link mask to be enabled by setting ctx->linkMask = m_linkMask
     * - if (m_ePhyMode == NVMEDIA_ICP_CSI_CPHY_MODE)
     *  - set ctx->phyMode = CDI_MAX96712_PHY_MODE_CPHY
     *  .
     * - else
     *   - set ctx->phyMode = CDI_MAX96712_PHY_MODE_DPHY
     *   .
     *  - set ctx->defaultResetAll = m_resetAll
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    SIPLStatus SetMAX96712Ctx();

    /**
     * @brief setup configuration
     * - acquire new deserializer driver handle, saved in m_pCDIDriver, call GetMAX96712NewDriver
     * - if failed (got NULL)
     *  - error exit  NVSIPL_STATUS_ERROR
     *  .
     * - initialize context structure
     *
     * @param[in] deserInfoObj pointer to deserializer information
     * @param[in] params unused
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    virtual SIPLStatus DoSetConfig(DeserInfo const* const deserInfoObj, DeserializerParams *const params) override;

    /**
     * @brief initializes deserializer
     * - check that deserializer is present on platform:  nvmStatus = MAX96712CheckPresence(m_upCDIDevice.get())
     *  - if not - exit with error: ConvertNvMediaStatus(nvmStatus)
     *  .
     * - Set deserializer defaults:  MAX96712SetDefaults(m_upCDIDevice.get())
     * - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     * - Get deserializer revision by
     *  -  MAX96712ReadParameters(
     *   - m_upCDIDevice.get(),
     *   - CDI_READ_PARAM_CMD_MAX96712_REV_ID,
     *   - sizeof(int32_t),
     *   - &readParamsMAX96712);
     *   .
     *  .
     * - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     * - set m_eRevision = readParamsMAX96712.revision
     * - prepare parameter to be set in writeParamsMAX96712 :
     *  - writeParamsMAX96712.MipiSettings.mipiSpeed = static_cast<uint8_t>(m_uMipiSpeed / 100000U);
     *  - writeParamsMAX96712.MipiSettings.phyMode = (m_ePhyMode == NVMEDIA_ICP_CSI_CPHY_MODE) ? CDI_MAX96712_PHY_MODE_CPHY : CDI_MAX96712_PHY_MODE_DPHY;
     *  .
     * - write parameter by:
     *  - nvmStatus = MAX96712WriteParameters(
     *   - m_upCDIDevice.get(),
     *   - CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI,
     *   - sizeof(writeParamsMAX96712.MipiSettings),
     *   - &writeParamsMAX96712);
     *   .
     *  .
     * - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     * - Disable BACKTOP by:
     *  - nvmStatus = MAX96712SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_MAX96712_DISABLE_BACKTOP)
     *  .
     * - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    virtual SIPLStatus DoInit() override;

    /**
     * @brief Device specific implementation for device start operation.
     * - Check CSIPLL lock  by:
     *  - nvmStatus = MAX96712SetDeviceConfig(
     *   - m_upCDIDevice.get(),
     *   - CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK)
     *   .
     *  .
     * -  Trigger the initial deskew by:
     *  - if ((m_ePhyMode == NVMEDIA_ICP_CSI_DPHY_MODE) and (m_uMipiSpeed >= 1500000U))
     *   - nvmStatus = MAX96712SetDeviceConfig(
     *    - m_upCDIDevice.get(),
     *    - CDI_CONFIG_MAX96712_TRIGGER_DESKEW)
     *    .
     *   - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     *   .
     *  ,
     * - check link lock by  CheckLinkLock(m_linkMask)
     * - if not locked (error)
     *  -  nvmStatus = MAX96712OneShotReset(
     *   - NVMEDIA_STATUS_OK,
     *   - m_upCDIDevice.get(),
     *   - (static_cast<LinkMAX96712>(m_linkMask)))
     *  - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     *  - check link lock by  CheckLinkLock(m_linkMask)
     *  - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     *  .
     * - Enable BACKTOP by:
     *  - nvmStatus = MAX96712SetDeviceConfig(
     *   - m_upCDIDevice.get(),
     *   - CDI_CONFIG_MAX96712_ENABLE_BACKTOP)
     *   .
     *  - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     *  - Check & Clear if ERRB set by reading param readParamsMAX96712:
     *   - nvmStatus = MAX96712ReadParameters(
     *    - m_upCDIDevice.get(),
     *    - CDI_READ_PARAM_CMD_MAX96712_ERRB,
     *    - sizeof(readParamsMAX96712.ErrorStatus),
     *    - &readParamsMAX96712);
     *    .
     *   .
     *  - if failed - exit error ConvertNvMediaStatus(nvmStatus)
     * .
     *
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    virtual SIPLStatus DoStart() override;

    /**
     * @brief Device specific implementation for device stop operation.
     * @retval NVSIPL_STATUS_OK             On success.
     * @retval (SIPLStatus)                 Subclasses can override this
     */
    virtual SIPLStatus DoStop() override
    {
        return NVSIPL_STATUS_OK;
    };

    /**
     * @brief Set Context Init Value
     * - for each link index
     *  - set ctx->gmslMode[link_index] = CDI_MAX96712_GMSL_MODE_UNUSED
     *  - set ctx->longCables[link_index] = m_longCables[link_index]
     *
     * @param[in] ctx deserializer context
     * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
    void SetCtxInitValue_CNvMMax96712Fusa(ContextMAX96712 * const ctx) const;

    /**
     * @brief :Update GMSL Mode
     * - for each mode in m_ovLinkModes
     *  - if (item.linkIndex < MAX96712_MAX_NUM_LINK) i.e. is valid
     *   - if (item.elinkMode == LinkMode::LINK_MODE_GMSL2_6GBPS)
     *    - set ctx->gmslMode[item.linkIndex] = CDI_MAX96712_GMSL2_MODE_6GBPS
     *    .
     *   - else
     *    - if (item.elinkMode == LinkMode::LINK_MODE_GMSL2_3GBPS)
     *     - set ctx->gmslMode[item.linkIndex] = CDI_MAX96712_GMSL2_MODE_3GBPS
     *     .
     *    - else
     *     - error exit NVSIPL_STATUS_NOT_SUPPORTED
     *    .
     *   .
     *  - else
     *   - error exit NVSIPL_STATUS_NOT_SUPPORTED
     *   .
     * .
     *
     * @param[in] ctx deserilizer context
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    SIPLStatus UpdateGmslMode_CNvMMax96712Fusa(ContextMAX96712 * const ctx) const;

    /**
     * @brief Update Context Configuration
     * - if m_I2CPort == 0U
     *  - set ctx->i2cPort = CDI_MAX96712_I2CPORT_0
     *  .
     * - else
     *  - if m_I2CPort == 1U
     *   - set ctx->i2cPort = CDI_MAX96712_I2CPORT_1
     *   .
     *  - else
     *   - if m_I2CPort == 2U
     *    - set ctx->i2cPort = CDI_MAX96712_I2CPORT_2
     *    .
     *   .
     * - if m_eInterface == either values:
     *  -
     *   - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_A
     *   - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_C
     *   - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_E
     *   - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_G
     *   -
     *  - if ctx->i2cPort == CDI_MAX96712_I2CPORT_0
     *   - set ctx->i2cPort = CDI_MAX96712_TXPORT_PHY_C
     *   .
     *  - else
     *   - set ctx->i2cPort = CDI_MAX96712_TXPORT_PHY_E
     *   .
     *  - set ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_4x2
     * - else
     *  - if m_eInterface == either values:
     *   -
     *    - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_B
     *    - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_D
     *    - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_D
     *    - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_H
     *    -
     *   - if ctx->i2cPort == CDI_MAX96712_I2CPORT_0
     *    - set ctx->i2cPort = CDI_MAX96712_TXPORT_PHY_D
     *    .
     *   - else
     *    - set ctx->i2cPort = CDI_MAX96712_TXPORT_PHY_F
     *    .
     *   - set ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_4x2
     *  - else
     *   - if m_eInterface == either values:
     *    -
     *     - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_AB
     *     - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_CD
     *     - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_EF
     *     - NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_GH
     *     -
     *    - if ctx->i2cPort == CDI_MAX96712_I2CPORT_0
     *     - set ctx->i2cPort = CDI_MAX96712_TXPORT_PHY_D
     *     .
     *    - else
     *     - set ctx->i2cPort = CDI_MAX96712_TXPORT_PHY_E
     *     .
     *    - set ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_2x4
     *    .
     *   - else
     *    - error exit NVSIPL_STATUS_NOT_SUPPORTED
     *    .
     *   .
     *
     * @param[in] ctx deserializer context
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    SIPLStatus UpdateCtxConfiguration_CNvMMax96712Fusa(ContextMAX96712 * const ctx) const;

    /**
     * @brief Update Mipi Speed
     * - if PHY mode is NVMEDIA_ICP_CSI_DPHY_MODE
     *  - if MIPI outout mode is CDI_MAX96712_MIPI_OUT_2x4
     *   - set m_uMipiSpeed = m_dphyRate[X4_CSI_LANE_CONFIGURATION]
     *   .
     *  - else
     *   - set m_uMipiSpeed = m_dphyRate[X2_CSI_LANE_CONFIGURATION]
     *   .
     *  .
     * - else
     *  - if PHY mode is NVMEDIA_ICP_CSI_CPHY_MODE
     *   - set m_uMipiSpeed = m_cphyRate[X4_CSI_LANE_CONFIGURATION]
     *   .
     *  - else
     *   - set m_uMipiSpeed = m_cphyRate[X2_CSI_LANE_CONFIGURATION]
     *   .
     *  - else
     *   - error exit NVSIPL_STATUS_NOT_SUPPORTED
     *
     * @param[in] ctx context setup for deserializer
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    SIPLStatus UpdateMipiSpeed_CNvMMax96712Fusa(const ContextMAX96712 * const ctx);

    /**
     * @brief update transmit port
     * - verify m_TxPort is valid
     * - if m_TxPort == 0
     *  - set ctx->txPort = CDI_MAX96712_TXPORT_PHY_C
     *  .
     * - else
     *  - if m_TxPort == 1
     *   - set ctx->txPort = CDI_MAX96712_TXPORT_PHY_D
     *   .
     *  - else
     *   - if m_TxPort == 2
     *    - set ctx->txPort = CDI_MAX96712_TXPORT_PHY_E
     *    .
     *   - else
     *    - if m_TxPort == 3
     *     - set ctx->txPort = CDI_MAX96712_TXPORT_PHY_F
     *     .
     *    - else
     *     - error exit NVSIPL_STATUS_NOT_SUPPORTED
     *
     * @param[in] ctx deserializer context
     * @return NVSIPL_STATUS_OK if done, otherwise an error code */
    SIPLStatus UpdateTxPort_CNvMMax96712Fusa(ContextMAX96712 * const ctx) const;

#if !NV_IS_SAFETY || SAFETY_DBG_OV
    /**
     * @brief Configure the pipeline to copy data in the deserializer
     * - if m_sDeserializerName is "MAX96712_Fusa_nv_camRecCfg_V1" or "MAX96722_Fusa_nv_camRecCfg_V1"
     *  - set ctx->cfgPipeCopy = CDI_MAX96712_CFG_PIPE_COPY_MODE_1
     *  .
     * - else if m_sDeserializerName is "MAX96712_Fusa_nv_camRecCfg_V2" or "MAX96722_Fusa_nv_camRecCfg_V2"
     *  - set ctx->cfgPipeCopy = CDI_MAX96712_CFG_PIPE_COPY_MODE_2
     * .
     *
     * @param[in] ctx deserializer context
     **/
    void SetMAX96712CfgPipelineCpy(ContextMAX96712 *ctx) noexcept;
#endif // !NV_IS_SAFETY || SAFETY_DBG_OV
private:
    //! Holds the revision of the MAX96712
    RevisionMAX96712 m_eRevision;

    /**
     * @brief Function to map SIPL log verbosity level to
     * C log verbosity level.
     *
     * @param[in] level SIPL Log trace level.
     * @return C log level.
     */
    C_LogLevel ConvertLogLevel(INvSIPLDeviceBlockTrace::TraceLevel const level);

#if !NV_IS_SAFETY || SAFETY_DBG_OV
    /*! Name of deserializer */
    std::string m_sDeserializerName;
#endif // !NV_IS_SAFETY || SAFETY_DBG_OV
};

} // end of namespace nvsipl
#endif /* CNVMMAX96712_FUSA_NV_HPP */
