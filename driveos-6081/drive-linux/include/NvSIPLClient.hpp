/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVSIPLCLIENT_HPP
#define NVSIPLCLIENT_HPP

#include "NvSIPLCommon.hpp"
#include "NvSIPLISPStructs.hpp"

#include "NvSIPLCDICommon.h"

#include "nvscisync.h"
#include "nvscistream.h"

#include <cstdint>
#include <string>
#include <memory>

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: Client Interface - @ref NvSIPLClient_API </b>
 *
 */

/** @defgroup NvSIPLClient_API NvSIPL Client
 *
 * @brief Provides interfaces to retrieve the output of the SIPL Pipeline Manager.
 *
 * @ingroup NvSIPLCamera_API
 */

namespace nvsipl
{
/** @ingroup NvSIPLClient_API
 * @{
 */

/** @class INvSIPLClient NvSIPLClient.hpp
 *
 * @brief Defines the public data structures
 * and describes the interfaces for @ref NvSIPLClient_API.
 *
 */
class INvSIPLClient
{
public:
    /** @brief Defines the metadata associated with the image. */
    struct ImageMetaData
    {
        /** Holds the TSC timestamp of the end of frame for capture. */
        uint64_t frameCaptureTSC;
        /** Holds the TSC timestamp of the start of frame for capture. */
        uint64_t frameCaptureStartTSC;
        /** Holds the parsed embedded data frame number of exposures info for the captured frame.*/
        uint32_t numExposures;
        /** Holds the parsed embedded data sensor exposure info for the captured frame. */
        DevBlkCDIExposure sensorExpInfo;
        /** Holds the parsed embedded data sensor white balance info for the captured frame. */
        DevBlkCDIWhiteBalance sensorWBInfo;
        /** Holds the parsed embedded data illumination info for the captured frame. */
        DevBlkCDIIllumination illuminationInfo;
        /** Holds the parsed embedded data sensor PWL info for the captured frame. */
        DevBlkCDIPWL sensorPWLInfo;
        /** Holds the parsed embedded data sensor crc info for the captured frame. */
        DevBlkCDICRC sensorCRCInfo;
        /** Holds the parsed embedded data frame report info for the captured frame. */
        DevBlkCDIFrameReport sensorReportInfo;
        /** Holds the parsed embedded data sensor temperature info for the captured frame. */
        DevBlkCDITemperature sensorTempInfo;
        /** Holds the parsed embedded data frame sequence number info for the captured frame. */
        DevBlkCDIFrameSeqNum frameSeqNumInfo;
        /** Holds a flag indicating if the ISP bad pixel statistics are valid. */
        bool badPixelStatsValid;
        /** Holds the ISP bad pixel statistics for the previous ISP output frame. */
        NvSiplISPBadPixelStatsData badPixelStats;
        /** Holds the ISP bad pixel settings for the previous ISP output frame. */
        NvSiplISPBadPixelStats badPixelSettings;
        /** Holds parsed embedded data frame timestamp info for the captured frame. */
        DevBlkCDIFrameTimestamp frameTimestampInfo;

        /** Holds a flag indicating if the ISP Histogram statistics are valid. */
        bool histogramStatsValid[2];
        /** Holds the ISP Histogram statistics for the previous ISP output frame. */
        NvSiplISPHistogramStatsData histogramStats[2];

        /** Holds the ISP Histogram settings for the previous ISP output frame. */
        NvSiplISPHistogramStats histogramSettings[2];
        /** Holds a flag indicating if the ISP Local Average and Clipped statistics are valid. */
        bool localAvgClipStatsValid[2];
        /** Holds the ISP Local Average and Clipped statistics for the previous ISP output frame. */
        NvSiplISPLocalAvgClipStatsData localAvgClipStats[2];
        /** Holds the ISP Local Average and Clipped settings for the previous ISP output frame. */
        NvSiplISPLocalAvgClipStats localAvgClipSettings[2];
        /** Holds information on errors present in the embedded data.
         * The meaning of these values is determined by the driver. */
        int8_t errorFlag;
        /** Holds the control information. */
        NvSiplControlInfo controlInfo;

    };

     struct ImageEmbeddedData
     {
          /** Holds size of the top embedded data. */
          uint32_t embeddedBufTopSize;
          /** Holds size of the bottom embedded data. */
          uint32_t embeddedBufBottomSize;
          /** Holds pointer to the top embedded data. */
          uint8_t *embeddedBufTop;
          /** Holds pointer to the bottom embedded data. */
          uint8_t *embeddedBufBottom;
     };

    /** @class INvSIPLBuffer
     *
     * @brief Abstract interface for SIPL buffers.
     *
     */
    class INvSIPLBuffer
    {
     public:
        /** @brief Adds a reference.
         *
         * Adding a reference to the buffer ensures that this buffer is not
         * re-used by the producer of the buffer.
         *
         * @pre None.
         *
         * @usage
         * - Allowed context for the API call
         *   - Interrupt handler: No
         *   - Signal handler: No
         *   - Thread-safe: Yes, with the following conditions:
         *     - There is no active release operation that results in the buffer
         *       available for re-use.
         *   - Re-entrant: No
         *   - Async/Sync: Sync
         * - Required privileges: Yes, with the following conditions:
         *   - Grants: nonroot, allow
         *   - Abilities: public_channel
         *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
         *     NVIDIA DRIVE OS Safety Developer Guide
         * - API group
         *   - Init: No
         *   - Runtime: Yes
         *   - De-Init: No
         */
        virtual void AddRef() = 0;

        /** @brief Release a reference.
         *
         * Once the reference has been released as many times as the reference
         * was added, it implies that the user has finished working with the
         * buffer and the buffer is available for re-use by SIPL.
         *
         * @pre None.
         *
         * Release() originates the following SIPLStatus return values:
         * @retval NVSIPL_STATUS_OK  The releasing operation succeeds.
         * @retval NVSIPL_STATUS_BAD_ARGUMENT If the buffer is already released.
         * @retval SIPLStatus Other values of SIPLStatus are propagated.
         *
         * @usage
         * - Allowed context for the API call
         *   - Interrupt handler: No
         *   - Signal handler: No
         *   - Thread-safe: Yes
         *   - Re-entrant: No
         *   - Async/Sync: Sync
         * - Required privileges: Yes, with the following conditions:
         *   - Grants: nonroot, allow
         *   - Abilities: public_channel
         *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
         *     NVIDIA DRIVE OS Safety Developer Guide
         * - API group
         *   - Init: No
         *   - Runtime: Yes
         *   - De-Init: No
         */
        virtual SIPLStatus Release() = 0;

        /** @brief Add an NvSciSync prefence.
         *
         * Add an NvSciSync prefence to be used with the next ISP or ICP operation. This function
         * creates its own duplicate of the fence, so the caller must clear their copy of the fence
         * by calling @ref NvSciSyncFenceClear().
         *
         * ICP operations support up to 3 unique pre-fences per pipeline
         * ISP operations support up to 8 unique pre-fences per pipeline
         *
         * Users need to ensure that the number of unique pre-fences for ISP does not exceed 8 across
         * the all output buffers if using more than one ISP outputs for the pipeline.
         *
         * @pre None.
         *
         * @param[in] prefence Prefence to be added.
         *
         * AddNvSciSyncPrefence() originates the following SIPLStatus return values:
         * @retval NVSIPL_STATUS_OK  The function succeeds.
         * @retval NVSIPL_STATUS_ERROR If internal NvSci fence operations fail.
         * @retval SIPLStatus Other values of SIPLStatus are propagated.
         *
         * @usage
         * - Allowed context for the API call
         *   - Interrupt handler: No
         *   - Signal handler: No
         *   - Thread-safe: Yes
         *   - Re-entrant: No
         *   - Async/Sync: Sync
         * - Required privileges: Yes, with the following conditions:
         *   - Grants: nonroot, allow
         *   - Abilities: public_channel
         *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
         *     NVIDIA DRIVE OS Safety Developer Guide
         * - API group
         *   - Init: No
         *   - Runtime: Yes
         *   - De-Init: No
         */
        virtual SIPLStatus AddNvSciSyncPrefence(const NvSciSyncFence &prefence) = 0;

        /** @brief Retrieve the latest NvSciSync EOF fence.
         *
         * Retrieve the buffer's latest NvSciSync EOF fence associated with the engine's set
         * NvSciSync EOF object. The caller must clear the returned fence by calling
         * @ref NvSciSyncFenceClear().
         *
         * @pre None.
         *
         * @param[out] postfence EOF fence being returned.
         *
         * GetEOFNvSciSyncFence() originates the following SIPLStatus return values:
         * @retval NVSIPL_STATUS_OK  The function succeeds.
         * @retval NVSIPL_STATUS_ERROR If internal NvSci fence operations fail.
         * @retval SIPLStatus Other values of SIPLStatus are propagated.
         *
         * @usage
         * - Allowed context for the API call
         *   - Interrupt handler: No
         *   - Signal handler: No
         *   - Thread-safe: Yes
         *   - Re-entrant: Yes
         *   - Async/Sync: Sync
         * - Required privileges: Yes, with the following conditions:
         *   - Grants: nonroot, allow
         *   - Abilities: public_channel
         *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
         *     NVIDIA DRIVE OS Safety Developer Guide
         * - API group
         *   - Init: No
         *   - Runtime: Yes
         *   - De-Init: No
         */
        virtual SIPLStatus GetEOFNvSciSyncFence(NvSciSyncFence *const postfence) = 0;
    };

    /** @class INvSIPLNvMBuffer
     *
     * @brief Describes a SIPL buffer containing an @ref NvSciBufObj.
     */
    class INvSIPLNvMBuffer : public INvSIPLBuffer
    {
     public:
        /** @brief Gets a handle to @ref NvSciBufObj.
         *
         * @pre None.
         *
         * @returns A @ref NvSciBufObj.
         *
         * @usage
         * - Allowed context for the API call
         *   - Interrupt handler: No
         *   - Signal handler: No
         *   - Thread-safe: Yes
         *   - Re-entrant: Yes
         *   - Async/Sync: Sync
         * - Required privileges: Yes, with the following conditions:
         *   - Grants: nonroot, allow
         *   - Abilities: public_channel
         *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
         *     NVIDIA DRIVE OS Safety Developer Guide
         * - API group
         *   - Init: No
         *   - Runtime: Yes
         *   - De-Init: No
         */
        virtual NvSciBufObj GetNvSciBufImage() const = 0;

        /** @brief Gets an @ref nvsipl::INvSIPLClient::ImageMetaData
         * associated with @ref NvSciBufObj
         * @returns A const reference to @ref nvsipl::INvSIPLClient::ImageMetaData.
         *
         * @pre None.
         *
         * @usage
         * - Allowed context for the API call
         *   - Interrupt handler: No
         *   - Signal handler: No
         *   - Thread-safe: Yes
         *   - Re-entrant: Yes
         *   - Async/Sync: Sync
         * - Required privileges: Yes, with the following conditions:
         *   - Grants: nonroot, allow
         *   - Abilities: public_channel
         *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
         *     NVIDIA DRIVE OS Safety Developer Guide
         * - API group
         *   - Init: No
         *   - Runtime: Yes
         *   - De-Init: No
         */
        virtual ImageMetaData const& GetImageData() const = 0;

        /** @brief Gets an @ref nvsipl::INvSIPLClient::ImageEmbeddedData
         * this is the RAW data associated with the captured image from the sensor before
         * it is parsed to ImageMetaData structure.
         *
         * @pre None.
         *
         * @returns A const reference to @ref nvsipl::INvSIPLClient::ImageEmbeddedData.
         *
         * @usage
         * - Allowed context for the API call
         *   - Interrupt handler: No
         *   - Signal handler: No
         *   - Thread-safe: Yes
         *   - Re-entrant: Yes
         *   - Async/Sync: Sync
         * - Required privileges: Yes, with the following conditions:
         *   - Grants: nonroot, allow
         *   - Abilities: public_channel
         *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
         *     NVIDIA DRIVE OS Safety Developer Guide
         * - API group
         *   - Init: No
         *   - Runtime: Yes
         *   - De-Init: No
         */
        virtual ImageEmbeddedData const& GetImageEmbeddedData() const = 0;

    };

    /** @brief Describes a client of the pipeline. */
    struct ConsumerDesc
    {
        /** @brief Defines the types of the SIPL pipeline output. */
        enum class OutputType
        {
            ICP,  /**< Indicates the unprocessed output of the image sensor. */
            ISP0, /**< Indicates the first output of ISP. */
            ISP1, /**< Indicates the second output of ISP. */
            ISP2, /**< Indicates the third output of ISP. */
        };
    };
};

/** @} */

} // namespace nvsipl

#endif // NVSIPLCLIENT_HPP
