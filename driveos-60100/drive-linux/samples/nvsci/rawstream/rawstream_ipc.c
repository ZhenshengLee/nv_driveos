/*
 * Copyright (c) 2020-2022 NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */

#include "rawstream.h"
#include <sys/neutrino.h>
#define TEST_EVENT_CODE (0x30) /* 1 byte */

// Initialize one end of named communcation channel
NvSciError ipcInit(const char* endpointName, IpcWrapper* ipcWrapper)
{
    NvSciError err = NvSciError_Success;
    int32_t chid;
    int32_t coid;

    // Open named endpoint
    err = NvSciIpcOpenEndpoint(endpointName, &ipcWrapper->endpoint);
    if (err != NvSciError_Success) {
        fprintf(stderr, "Unable to open endpoint %s (%x)\n",
                endpointName, err);
        goto fail;
    }

    // Create QNX channel for monitoring IPC
    chid = ChannelCreate_r(_NTO_CHF_UNBLOCK);
    if (0 > chid) {
        err = NvSciError_ResourceError;
        fprintf(stderr, "ChannelCreate_r failed (%d:%x)\n", chid, err);
        goto fail;;
    }
    ipcWrapper->chId = chid;

    // Connect QNX channel for monitoring IPC
    coid = ConnectAttach_r(0, 0, chid, _NTO_SIDE_CHANNEL, 0);
    if (0 > coid) {
        err = NvSciError_ResourceError;
        fprintf(stderr, "ConnectAttach_r failed (%d:%x)\n", coid, err);
        goto fail;
    }
    ipcWrapper->connId = coid;

    // Bind IPC events to QNX connection
    err = NvSciIpcSetQnxPulseParamSafe(ipcWrapper->endpoint,
                                       ipcWrapper->connId,
                                       SIGEV_PULSE_PRIO_INHERIT,
                                       TEST_EVENT_CODE);
    if (NvSciError_Success != err) {
        fprintf(stderr, "NvSciIpcSetQnxPulseParamSafe: failed (%x)\n",
                err);
        goto fail;;
    }

    // Retrieve endpoint info
    err = NvSciIpcGetEndpointInfo(ipcWrapper->endpoint, &ipcWrapper->info);
    if (NvSciError_Success != err) {
        fprintf(stderr, "Unable to retrieve IPC endpoint info (%x)", err);
        goto fail;
    }

    err = NvSciIpcResetEndpointSafe(ipcWrapper->endpoint);
    if (NvSciError_Success != err) {
        fprintf(stderr, "Unable to reset IPC endpoint (%x)", err);
    }

fail:
    return err;
}

// Clean up IPC when done
void ipcDeinit(IpcWrapper* ipcWrapper)
{
    NvSciError err = NvSciIpcCloseEndpointSafe(ipcWrapper->endpoint, false);
    if (NvSciError_Success != err) {
        fprintf(stderr, "NvSciIpcCloseEndpointSafe failed (%x)\n", err);
    }

    if (ipcWrapper->connId != 0) {
        (void)ConnectDetach_r(ipcWrapper->connId);
        ipcWrapper->connId = 0;
    }
    if (ipcWrapper->chId != 0) {
        (void)ChannelDestroy_r(ipcWrapper->chId);
        ipcWrapper->chId = 0;
    }
}

// Wait for an event on IPC channel
static NvSciError waitEvent(IpcWrapper* ipcWrapper, uint32_t value)
{
    struct _pulse pulse;
    uint32_t event = 0;
    NvSciError err;

    while (true) {
        // Get pending IPC events
        err = NvSciIpcGetEventSafe(ipcWrapper->endpoint, &event);
        if (NvSciError_Success != err) {
            fprintf(stderr, "NvSciIpcGetEventSafe failed (%x)\n", err);
            return err;
        }
        // Return if event is the kind we're looking for
        if (0U != (event & value)) {
            break;
        }

        // Wait for pulse indicating new event
        int32_t ret =
            NvSciIpcWaitEventQnx(ipcWrapper->chId,
                                 NVSCIIPC_INFINITE_WAIT,
                                 sizeof(pulse), &pulse);
        if (0 > ret) {
            err = NvSciError_ResourceError;
            fprintf(stderr, "NvSciIpcWaitEventQnx failed (%d:%x)\n", ret, err);
            return err;
        }
        if (TEST_EVENT_CODE != pulse.code) {
            fprintf(stderr, "Invalid pulse %d\n", pulse.code);
            return NvSciError_NvSciIpcUnknown;
        }
    }

    return NvSciError_Success;
}

// Send a message over IPC
NvSciError ipcSend(IpcWrapper* ipcWrapper, const void* buf, const size_t size)
{
    NvSciError err = NvSciError_Success;
    bool done = false;
    uint32_t bytes;

    // Loop until entire message sent
    while (done == false) {

        // Wait for room in channel to send a message
        err = waitEvent(ipcWrapper, NV_SCI_IPC_EVENT_WRITE);
        if (NvSciError_Success != err) {
            goto fail;
        }

        assert(size <= UINT32_MAX);

        // Send as much of the message as we can
        err = NvSciIpcWriteSafe(ipcWrapper->endpoint, buf, (uint32_t)size,
                                &bytes);
        if (NvSciError_Success != err) {
            fprintf(stderr, "IPC write failed (%x)\n", err);
            goto fail;
        }

        // For this simple sample, we just fail if the entire message wasn't
        //   sent. Could instead retry to send the rest.
        if (size != (size_t)bytes) {
            fprintf(stderr, "Failed to send entire message (%u < %zu)\n",
                    bytes, size);
            err = NvSciError_NvSciIpcUnknown;
            goto fail;
        }
        done = true;
    }

fail:
    return err;
}

// Receive a message over IPC
NvSciError ipcRecvFill(IpcWrapper* ipcWrapper, void* buf, const size_t size)
{
    NvSciError err = NvSciError_Success;
    bool done = false;
    uint32_t bytes;

    // Loop until entire message received
    while (done == false) {

        // Wait for incoming data
        err = waitEvent(ipcWrapper, NV_SCI_IPC_EVENT_READ);
        if (NvSciError_Success != err) {
            goto fail;
        }

        assert(size <= UINT32_MAX);

        // Read as much of the message as we can
        err = NvSciIpcReadSafe(ipcWrapper->endpoint, buf, (uint32_t)size,
                               &bytes);
        if (NvSciError_Success != err) {
            fprintf(stderr, "IPC read failed (%x)\n", err);
            goto fail;
        }

        // For this simple sample, we just fail if the entire message wasn't
        //   read. Could instead retry to receive the rest.
        if (size != (size_t)bytes) {
            fprintf(stderr, "Failed to read entire message (%u < %zu)\n",
                    bytes, size);
            err = NvSciError_NvSciIpcUnknown;
            goto fail;
        }
        done = true;
    }

fail:
    return err;
}
