/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * @file NvFsiCom.h
 * @brief  NvFsiCom APIs are declared in this file
 */

#ifndef NVFSICOM_H
#define NVFSICOM_H

#include <NvFsiComTypes.h>

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief API to get application dt node for FSI communication. dtb is read in
 *        this API
 *
 *
 * @param[in]  NumChannel - Number of channel that Application has configured in
 *                          dt for FSI communication.
 * @param[in]  compat - compatibility string which will be used to get dt node
 *
 * @param[out] pHandle - pointer to array of handlers.
 *                       A handler will be provided, which will be used to
 *                       identify channel used by Application for peer to peer
 *                       communication
 *
 * @return :: -EOK -      No Error
 *            -ENOTSUP - if NumChannel is more than configured channels in dt
 *            -EINVAL -  dt node is not found in dt or invalid input param
 *
 * @pre Application should configure a dt node to list the channel Ids which
 *      it will use
 *
 *   example Application dt node -:
 *
 *   FsiComAppChConfApp1{
 *   compatible = "nvidia,tegra-fsicom-qnx-sampleApp1";
 *   status = "okay";
 *   channelid_list = <0 1>;
 *   };
 *
 *   Application should configure channel, Number of Frames and Frame size in
 *   FsiComIvc dt node
 *
 *   Sample channel config -:
 *   channel_0{
 *   frame-count = <4>;
 *   frame-size = <64>;
 *   };
 *
 * @usage
 * - Allowed context for the API call
 *   - Cancellation point: No
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync mode: Sync
 * - Required privileges: No
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
int32_t NvFsiComGetNodeInfo(NvFsiComHandle_t* pHandle, uint8_t NumChannel, const char *compat);

/**
 * @brief API to Initialise all FsiCom communication channels and establish
 *        connection with device. Connection info is saved in global variable
 *        in library scope.
 *
 *
 * @param[in]  *pHandle  - pointer to array of FsiCom channel handlers array
 * @param[in]  NumChannel- Number of channels/handlers in array
 *
 * @return :: EOK      - No Error
 *            -ENODEV  - Error in opening device
 *            -ECOMM   - Error in communication with device
 *            -EBUSY   - FsiCom channel already initialized
 *            -EPERM   - Operation not permitted
 *
 * @pre NvFsiComGetNodeInfo should be called before call this API
 *
 * @usage
 * - Allowed context for the API call
 *   - Cancellation point: No
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync mode: Sync
 * - Required privileges:  -A pathspace,public_channel 
 *                         -A nonroot,allow,able=NvMap/Interfaces:1-2
 *                         -A nonroot,allow,able=NvMap/Interfaces:17-19
 *                         -A nonroot,allow,able=SMMU/SID:<Stream Id>
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */

int32_t NvFsiComInit(NvFsiComHandle_t* pHandle, uint8_t NumChannel);

/**
 * @brief API to send data frames of pre defined size to FSI
 *
 *
 * @param[in]  *pHandle  - pointer to handle of FsiCom channel on which data is
 *                         to be sent
 * @param[in]  *buff -     pointer to data which need to be sent
 *
 * @param[in]  size -     size of buff which should be same as maximum payload size expected by the application
 *
 * @param[out] *bytes - Number of bytes which were successfully written
 *
 * @return :: EOK         - No Error
 *            -EMSGSIZE   - Inappropriate message buffer length in device
 *            -ECOMM      - Error in communication with device
 *            -ECONNRESET - Device init is in progress
 *            -EINVAL     - invlid input param
 *            -ENOMEM     - Queue is full, no data can be written
 *            -EOVERFLOW  - Channel header is corrupted
 *
 * @pre NvFsiComInit should be successfully called before calling this API
 *
 * @usage
 * - Allowed context for the API call
 *   - Cancellation point: Yes
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync mode: Sync
 * - Required privileges: -A nonroot,allow,able=NvFsiCom/Channel_tx:<Channel Id>
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
int32_t  NvFsiComWrite(NvFsiComHandle_t* pHandle, void* buff, uint32_t size, uint32_t* bytes );

/**
 * @brief This API waits for any event from device. This event indicate that
 *        data is available on one of the channel which is initialise by the
 *        Application
 *
 * @param[out]  *Index - Index on which data is available to read
 *               if data is available to read on multiple indexes then
 *               API will unblock multiple times.
 *
 * @return :: EOK         - No Error
 *            -EFAULT     - Failed to receive an event
 *            -EINTR      - The call was interrupted by Signal
 *            -ETIMEDOUT  - A Kernel timeout unblocked the call
 *            -ECANCELED  - Deinit request received
 *
 * @pre NvFsiComInit should be successfully called before calling this API.
 *      User should call this API in endless loop. Thread will be RECEIVE-blocked
 *      state until a message arrives.
 *      During Deinit NvFsiComWaitForEvent will return with -ECANCELED.
 *
 * @usage
 * - Allowed context for the API call
 *   - Cancellation point: No
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync mode: Sync
 * - Required privileges:  -A pathspace,public_channel
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
int32_t NvFsiComWaitForEvent(NvFsiHandleIndex* Index);

/**
 * @brief API to receive data frames of pre defined size from FSI
 *
 *
 * @param[in]  *pHandle - pointer to handle of FsiCom channel from which data is
 *                         to be read
 * @param[in] *buff - pointer for buffer where received data will be copied
 * @param[in] size - size of buff which should be same as maximum payload size
 *                    expected by the application
 *
 * @param[out]  *bytes - Number of bytes which were successfully read
 *
 * @return :: EOK         - No Error
 *            -ECONNRESET - Device init is in progress
 *            -EINVAL     - invalid input param
 *            -ENOMEM     - Receive Queue is empty
 *            -ENODATA    - No data to read
 *            -EOVERFLOW  - Channel header is corrupted, received payload length
 *                          is greater than size
 *
 * @pre NvFsiComInit should be successfully called before calling this API
 *      It is recommended to call this API after NvFsiComWaitForEvent unblocks
 *
 * @usage
 * - Allowed context for the API call
 *   - Cancellation point: Yes
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync mode: Sync
 * - Required privileges: -A nonroot,allow,able=NvFsiCom/Channel_rx:<Channel Id>
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
int32_t NvFsiComRead(NvFsiComHandle_t* pHandle, void* buff, uint32_t size, uint32_t* bytes );

/**
 * @brief API to DeInitialise all FsiCom communication channels.
 *
 *
 * @param[in]  *pHandle  - pointer to array of FsiCom channel handlers array
 * @param[in] NumChannel- Number of channels/handlers in array
 *
 * @return :: EOK      - No Error
 *            -ENODEV  - Error in opening device
 *            -ECOMM   - Error in communication with device
 *            -EINVAL  - Invalid argument
 *
 * @pre NvFsiComInit should be called before call this API
 *
 * @usage
 * - Allowed context for the API call
 *   - Cancellation point: No
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync mode: Sync
 * - Required privileges:  -A pathspace,public_channel 
 *                         -A nonroot,allow,able=NvMap/Interfaces:1-2
 *                         -A nonroot,allow,able=NvMap/Interfaces:17-19
 *                         -A nonroot,allow,able=SMMU/SID:<Stream Id>
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 */

int32_t NvFsiComDeinit(NvFsiComHandle_t* pHandle, uint8_t NumChannel);

#ifdef __cplusplus
 }
#endif

#endif
