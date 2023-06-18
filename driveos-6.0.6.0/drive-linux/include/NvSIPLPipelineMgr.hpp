/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVSIPLPIPELINEMGR_HPP
#define NVSIPLPIPELINEMGR_HPP

#include "NvSIPLCommon.hpp"
#include "NvSIPLClient.hpp"
#include "NvSIPLPlatformCfg.hpp"
#include "NvSIPLInterrupts.hpp"

#include <cstdint>
#include <vector>

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: Pipeline Manager -
 *     @ref NvSIPLPipelineMgr </b>
 *
 */

namespace nvsipl
{
/** @defgroup NvSIPLPipelineMgr NvSIPL Pipeline Manager
 *
 * @brief Programs Video Input (VI) and Image Signal Processor (ISP)
 *  hardware blocks to create image processing pipelines
 *  for each sensor.
 *
 * @ingroup NvSIPLCamera_API
 * @{
 */

/**
 *
 * @class NvSIPLPipelineNotifier
 *
 * @brief Describes the interfaces of the SIPL pipeline notification handler.
 *
 * This class defines data structures and interfaces that must be implemented by
 * a SIPL pipeline notification handler.
 */
class NvSIPLPipelineNotifier
{
    /** Indicates the maximum number of gpio indices. */
    static constexpr uint32_t MAX_DEVICE_GPIOS {8U};

public:
    /** @brief Defines the events of the image processing pipeline and the device block. */
    enum NotificationType
    {
        /**
         * Pipeline event, indicates ICP processing is finished.
         * @note Only eNotifType, uIndex & frameCaptureTSC are valid in NotificationData for this event.
         */
        NOTIF_INFO_ICP_PROCESSING_DONE = 0,

        /**
         * Pipeline event, indicates ISP processing is finished.
         * @note Only eNotifType, uIndex & frameCaptureTSC are valid in NotificationData for this event.
         */
        NOTIF_INFO_ISP_PROCESSING_DONE = 1,

        /**
         * Pipeline event, indicates auto control processing is finished.
         * @note Only eNotifType, uIndex & frameCaptureTSC are valid in NotificationData for this event.
         */
        NOTIF_INFO_ACP_PROCESSING_DONE = 2,

        /**
         * Pipeline event, indicates CDI processing is finished.
         * @note This event is sent only if the Auto Exposure and Auto White Balance algorithm produces
         * new sensor settings that need to be updated in the image sensor.
         * @note Only eNotifType, uIndex & frameCaptureTSC are valid in NotificationData for this event.
         */
        NOTIF_INFO_CDI_PROCESSING_DONE = 3,

        /**
         * Pipeline event, indicates image authentication success.
         * @note Only eNotifType, uIndex & frameSeqNumber are valid in NotificationData for this event.
         */
        NOTIF_INFO_ICP_AUTH_SUCCESS = 4,

        /**
         * Pipeline event, indicates pipeline was forced to drop a frame due to a slow consumer or system issues.
         * @note Only eNotifType & uIndex are valid in NotificationData for this event.
         */
        NOTIF_WARN_ICP_FRAME_DROP = 100,

        /**
         * Pipeline event, indicates a discontinuity was detected in parsed embedded data frame sequence number.
         * @note Only eNotifType & uIndex are valid in NotificationData for this event.
         */
        NOTIF_WARN_ICP_FRAME_DISCONTINUITY = 101,

        /**
         * Pipeline event, indicates occurrence of timeout while capturing.
         * @note Only eNotifType & uIndex are valid in NotificationData for this event.
         */
        NOTIF_WARN_ICP_CAPTURE_TIMEOUT = 102,

        /**
         * Pipeline event, indicates ICP bad input stream.
         * @note Only eNotifType & uIndex are valid in NotificationData for this event.
         */
        NOTIF_ERROR_ICP_BAD_INPUT_STREAM = 200,

        /**
         * Pipeline event, indicates ICP capture failure.
         * @note Only eNotifType & uIndex are valid in NotificationData for this event.
         */
        NOTIF_ERROR_ICP_CAPTURE_FAILURE = 201,

        /**
         * Pipeline event, indicates embedded data parsing failure.
         * @note Only eNotifType & uIndex are valid in NotificationData for this event.
         */
        NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE = 202,

        /**
         * Pipeline event, indicates ISP processing failure.
         * @note Only eNotifType & uIndex are valid in NotificationData for this event.
         */
        NOTIF_ERROR_ISP_PROCESSING_FAILURE = 203,

        /**
         * Pipeline event, indicates auto control processing failure.
         * @note Only eNotifType & uIndex are valid in NotificationData for this event.
         */
        NOTIF_ERROR_ACP_PROCESSING_FAILURE = 204,

        /**
         * Pipeline event, indicates CDI set sensor control failure.
         * @note Only eNotifType & uIndex are valid in NotificationData for this event.
         */
        NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE = 205,

        /**
         * Device block event, indicates a deserializer failure.
         * @note Only eNotifType & gpioIdxs valid in NotificationData for this event.
         */
        NOTIF_ERROR_DESERIALIZER_FAILURE = 207,

        /**
         * Device block event, indicates a serializer failure.
         * @note Only eNotifType, uLinkMask & gpioIdxs are valid in NotificationData for this event.
         */
        NOTIF_ERROR_SERIALIZER_FAILURE = 208,

        /**
         * Device block event, indicates a sensor failure.
         * @note Only eNotifType, uLinkMask & gpioIdxs are valid in NotificationData for this event.
         */
        NOTIF_ERROR_SENSOR_FAILURE = 209,

        /**
         * Pipeline event, indicates isp process failure due to recoverable errors.
         * @note Only eNotifType & uIndex are valid in NotificationData for this event.
         */
        NOTIF_ERROR_ISP_PROCESSING_FAILURE_RECOVERABLE = 210,

        /**
         * Pipeline event, indicates image authentication failure.
         * @note Only eNotifType, uIndex & frameSeqNumber are valid in NotificationData for this event.
         */
        NOTIF_ERROR_ICP_AUTH_FAILURE = 211,

        /**
         * Pipeline and device block event, indicates an unexpected internal failure.
         * @note For pipeline event, only eNotifType & uIndex are valid in NotificationData for this event.
         * @note For device block event, only eNotifType is valid in NotificationData for this event.
         */
        NOTIF_ERROR_INTERNAL_FAILURE = 300,
    };

    /**
     * @brief Defines the notification data.
     * @note A few members are not valid for certain events, please see @ref NotificationType.
     */
    struct NotificationData
    {
        /** Holds the @ref NotificationType event type. */
        NotificationType eNotifType;
        /** Holds the ID of the pipeline. This is the same as the Sensor ID in PlatformCfg. */
        uint32_t uIndex;
        /** Holds the device block link mask. */
        uint8_t uLinkMask;
        /** Holds a sequence number of a captured frame. */
        uint64_t frameSeqNumber;
        /** Holds the TSC timestamp of the end of frame for capture. */
        uint64_t frameCaptureTSC;
        /** Holds the TSC timestamp of the start of frame for capture. */
        uint64_t frameCaptureStartTSC;
        /** Holds the Interrupt Code */
        InterruptCode intrCode;
        /** Holds the Interrupt Data */
        uint64_t intrData;
        /** Holds the GPIO indices. */
        uint32_t gpioIdxs[MAX_DEVICE_GPIOS];
        /** Holds the number of GPIO indices in the array. */
        uint32_t numGpioIdxs;
    };

    /** @brief Default destructor. */
    virtual ~NvSIPLPipelineNotifier(void) = default;
};

/**
 *
 * @class NvSIPLImageGroupWriter
 *
 * @brief Describes the interfaces of SIPL pipeline feeder.
 *
 * This class defines data structures and interfaces that must be implemented by
 * the SIPL pipeline feeder in case of ISP reprocess mode.
 *
 * In ISP reprocess mode, the user can feed unprocessed sensor output captured
 * during the data collection process and then process it through HW ISP.
 *
 * The user must have configured @ref NvSIPLCamera_API using an appropriate
 * PlatformCfg to be able
 * to use this mode.
 */
class NvSIPLImageGroupWriter
{
public:
    /** @brief Describes an unprocessed sensor output buffer. */
    struct RawBuffer
    {
        /** Holds an NvSciBufObj. */
        NvSciBufObj image;
        /** Holds the ID of the sensor in PlatformCfg. */
        uint32_t uIndex;
        /** Holds a flag to signal discontinuity for the current raw buffer from the previous one. */
        bool discontinuity;
        /** Holds a flag to signal that the pipeline should drop the current buffer. */
        bool dropBuffer;
        /** Holds the TSC timestamp of the end of frame for capture. */
        uint64_t frameCaptureTSC;
        /** Holds the TSC timestamp of the start of frame for capture. */
        uint64_t frameCaptureStartTSC;
    };

    /** @brief Populates the buffer with RAW data.
     *
     * The consumer's implementation overrides this method.
     * The method is called by SIPL pipeline thread.
     *
     * The feeder must populate the @ref RawBuffer with appropriate RAW data.
     *
     * @param[out] oRawBuffer   A reference to the @ref RawBuffer that
     *                           the function is to populate.
     *
     * @returns::SIPLStatus. The completion status of this operation. */
    virtual SIPLStatus FillRawBuffer(RawBuffer &oRawBuffer) = 0;

    /** @brief Default destructor. */
    virtual ~NvSIPLImageGroupWriter(void) = default;
};

/**
 * @brief Defines the camera pipeline configuration.
 *
 */
struct NvSIPLPipelineConfiguration
{
    /** <tt>true</tt> if the client wants capture output frames to be delivered */
    bool captureOutputRequested {false};

    /** <tt>true</tt> if the client wants frames to be delivered from the first ISP output */
    bool isp0OutputRequested {false};

    /** <tt>true</tt> if the client wants frames to be delivered from the second ISP output */
    bool isp1OutputRequested {false};

    /** <tt>true</tt> if the client wants frames to be delivered from the third ISP output */
    bool isp2OutputRequested {false};

    /** Holds a downscale and crop configuration. */
    NvSIPLDownscaleCropCfg downscaleCropCfg {};

    /**
     * Holds ISP statistics override parameters. ISP statistcis settings enabled
     * in @NvSIPLIspStatsOverrideSetting will override the statistics settings
     * provided in NITO
     */
    NvSIPLIspStatsOverrideSetting statsOverrideSettings {};

    /** Holds a pointer to an @ref NvSIPLImageGroupWriter. */
    NvSIPLImageGroupWriter* imageGroupWriter {nullptr};

    /** <tt>true</tt> if the client wants to disable the subframe feature */
    bool disableSubframe {false};
};

/**
 * @brief The interface to the frame completion queue.
 */
class INvSIPLFrameCompletionQueue
{
public:

    /**
     * @brief Retrieve the next item from the queue.
     *
     * The buffer returned will have a single reference that must be released by the client
     * when it has finished with the buffer. This is done by calling item->Release().
     *
     * @pre This function must be called after @ref INvSIPLCamera::Init() and before @ref INvSIPLCamera::Deinit().
     *
     * @param[out] item The item retrieved from the queue.
     * @param[in] timeoutUsec The timeout of the request, in microseconds.
     * If the queue is empty at the time of the call,
     * this method will wait up to @c timeoutUsec microseconds
     * for a new item to arrive in the queue and be returned.
     *
     * @retval NVSIPL_STATUS_OK if @c item has been successfully retrieved from the queue.
     * @retval NVSIPL_STATUS_TIMED_OUT if an item was not available within the timeout interval.
     * @retval NVSIPL_STATUS_EOF if the queue has been shut down.
     * In this case, no further calls can be made on the queue object.
     * @retval NVSIPL_STATUS_ERROR if a system error occurred.
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
    virtual SIPLStatus Get(INvSIPLClient::INvSIPLBuffer*& item,
                           size_t const timeoutUsec) = 0;

    /**
     * @brief Return the current queue length.
     *
     * @returns the number of elements currently in the queue.
     *
     * @pre This function must be called after @ref INvSIPLCamera::Init() and before @ref INvSIPLCamera::Deinit().
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
    virtual size_t GetCount() const = 0;

protected:

    INvSIPLFrameCompletionQueue() = default;
    virtual ~INvSIPLFrameCompletionQueue() = default;

private:

    INvSIPLFrameCompletionQueue(INvSIPLFrameCompletionQueue const& other) = delete;
    INvSIPLFrameCompletionQueue& operator=(INvSIPLFrameCompletionQueue const& rhs) = delete;
};

/**
 * @brief The interface to the notification queue.
 */
class INvSIPLNotificationQueue
{
public:

    /**
     * @brief Retrieve the next item from the queue.
     *
     * @pre This function must be called after @ref INvSIPLCamera::Init() and before @ref INvSIPLCamera::Deinit().
     *
     * @param[out] item The item retrieved from the queue.
     * @param[in] timeoutUsec The timeout of the request, in microseconds.
     * If the queue is empty at the time of the call,
     * this method will wait up to @c timeoutUsec microseconds
     * for a new item to arrive in the queue and be returned.
     *
     * @retval NVSIPL_STATUS_OK if @c item has been successfully retrieved from the queue.
     * @retval NVSIPL_STATUS_TIMED_OUT if an item was not available within the timeout interval.
     * @retval NVSIPL_STATUS_EOF if the queue has been shut down.
     * In this case, no further calls can be made on the queue object.
     * @retval NVSIPL_STATUS_ERROR if a system error occurred.
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
    virtual SIPLStatus Get(NvSIPLPipelineNotifier::NotificationData& item,
                           size_t const timeoutUsec) = 0;

    /**
     * @brief Return the current queue length.
     *
     * @pre This function must be called after @ref INvSIPLCamera::Init() and before @ref INvSIPLCamera::Deinit().
     *
     * @returns the number of elements currently in the queue.
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
    virtual size_t GetCount() const = 0;

protected:

    INvSIPLNotificationQueue() = default;
    virtual ~INvSIPLNotificationQueue() = default;

private:

    INvSIPLNotificationQueue(INvSIPLNotificationQueue const& other) = delete;
    INvSIPLNotificationQueue& operator=(INvSIPLNotificationQueue const& rhs) = delete;
};

/**
 * @brief This is the output structure for SetPipelineCfg().
 * It contains the queues used by the client to receive completed frames
 * and event notifications.
 */
struct NvSIPLPipelineQueues
{
    /**
     * The queue for completed capture frames.
     * Will be null if capture output was not requested.
     */
    INvSIPLFrameCompletionQueue* captureCompletionQueue {nullptr};

    /**
     * The queue for completed frames from the first ISP output.
     * Will be null if the first ISP output was not requested.
     */
    INvSIPLFrameCompletionQueue* isp0CompletionQueue {nullptr};

    /**
     * The queue for completed frames from the second ISP output.
     * Will be null if the second ISP output was not requested.
     */
    INvSIPLFrameCompletionQueue* isp1CompletionQueue {nullptr};

    /**
     * The queue for completed frames from the third ISP output.
     * Will be null if the third ISP output was not requested.
     */
    INvSIPLFrameCompletionQueue* isp2CompletionQueue {nullptr};

    /** The queue for event notifications. */
    INvSIPLNotificationQueue* notificationQueue {nullptr};
};

/**
 * @brief Holds the queues used by the client to receive device block event notifications.
 */
struct NvSIPLDeviceBlockQueues
{
    /** Queues for event notifications for each device block. */
    INvSIPLNotificationQueue* notificationQueue[MAX_DEVICEBLOCKS_PER_PLATFORM];
};

/** @} */

}  // namespace nvsipl


#endif // NVSIPLPIPELINEMGR_HPP
