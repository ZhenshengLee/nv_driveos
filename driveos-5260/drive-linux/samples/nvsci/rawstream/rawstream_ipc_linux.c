/*
 * Copyright (c) 2020 NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */

#include "rawstream.h"

// Initialize one end of named communcation channel
NvSciError ipcInit(const char* endpointName, IpcWrapper* ipcWrapper)
{
    NvSciError err = NvSciError_Success;

    // Open named endpoint
    err = NvSciIpcOpenEndpoint(endpointName, &ipcWrapper->endpoint);
    if (err != NvSciError_Success) {
        fprintf(stderr, "Unable to open endpoint %s (%x)\n",
                endpointName, err);
        goto fail;
    }

    // initialize IPC event notifier
    err = NvSciIpcGetLinuxEventFd(ipcWrapper->endpoint, &ipcWrapper->ipcEventFd);
    if (err != NvSciError_Success) {
        fprintf(stderr, "Unable to get Linux event fd (%x)\n", err);
        goto fail;
    }

    // Retrieve endpoint info
    err = NvSciIpcGetEndpointInfo(ipcWrapper->endpoint, &ipcWrapper->info);
    if (NvSciError_Success != err) {
        fprintf(stderr, "Unable to retrieve IPC endpoint info (%x)", err);
        goto fail;
    }

    NvSciIpcResetEndpoint(ipcWrapper->endpoint);

fail:
    return err;
}

// Clean up IPC when done
void ipcDeinit(IpcWrapper* ipcWrapper)
{
    NvSciIpcCloseEndpoint(ipcWrapper->endpoint);
}

// Wait for an event on IPC channel
static NvSciError waitEvent(IpcWrapper* ipcWrapper, uint32_t value)
{
    fd_set rfds;
    uint32_t event = 0;
    NvSciError err;

    while (true) {
        // Get pending IPC events
        err = NvSciIpcGetEvent(ipcWrapper->endpoint, &event);
        if (NvSciError_Success != err) {
            fprintf(stderr, "NvSciIpcGetEvent failed (%x)\n", err);
            return err;
        }
        // Return if event is the kind we're looking for
        if (0U != (event & value)) {
            break;
        }

        FD_ZERO(&rfds);
        FD_SET(ipcWrapper->ipcEventFd, &rfds);

        // Wait for signalling indicating new event
        if (select(ipcWrapper->ipcEventFd + 1, &rfds, NULL, NULL, NULL) < 0) {
            // select failed
            return NvSciError_ResourceError;
        }
        if(!FD_ISSET(ipcWrapper->ipcEventFd, &rfds)) {
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
    int32_t bytes;

    // Loop until entire message sent
    while (done == false) {

        // Wait for room in channel to send a message
        err = waitEvent(ipcWrapper, NV_SCI_IPC_EVENT_WRITE);
        if (NvSciError_Success != err) {
            goto fail;
        }

        // Send as much of the message as we can
        err = NvSciIpcWrite(ipcWrapper->endpoint, buf, size, &bytes);
        if (NvSciError_Success != err) {
            fprintf(stderr, "IPC write failed (%x)\n", err);
            goto fail;
        }

        // For this simple sample, we just fail if the entire message wasn't
        //   sent. Could instead retry to send the rest.
        if (size != (size_t)bytes) {
            fprintf(stderr, "Failed to send entire message (%d < %zu)\n",
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
    int32_t bytes;

    // Loop until entire message received
    while (done == false) {

        // Wait for incoming data
        err = waitEvent(ipcWrapper, NV_SCI_IPC_EVENT_READ);
        if (NvSciError_Success != err) {
            goto fail;
        }

        // Read as much of the message as we can
        err = NvSciIpcRead(ipcWrapper->endpoint, buf, size, &bytes);
        if (NvSciError_Success != err) {
            fprintf(stderr, "IPC read failed (%x)\n", err);
            goto fail;
        }

        // For this simple sample, we just fail if the entire message wasn't
        //   read. Could instead retry to receive the rest.
        if (size != (size_t)bytes) {
            fprintf(stderr, "Failed to read entire message (%d < %zu)\n",
                    bytes, size);
            err = NvSciError_NvSciIpcUnknown;
            goto fail;
        }
        done = true;
    }

fail:
    return err;
}
