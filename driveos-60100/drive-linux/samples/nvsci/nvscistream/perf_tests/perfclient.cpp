//! \file
//! \brief NvSciStream test client declaration.
//!
//! \copyright
//! SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//! SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//!
//! NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
//! property and proprietary rights in and to this material, related
//! documentation and any modifications thereto. Any use, reproduction,
//! disclosure or distribution of this material and related documentation
//! without an express license agreement from NVIDIA CORPORATION or
//! its affiliates is strictly prohibited.

#include <thread>
#include "nvplayfair.h"
#include "perfclient.h"
#include "poolhandler.h"

extern TestArg testArg;

PerfClient::PerfClient(NvSciBufModule buf,
                       NvSciSyncModule sync):
    bufModule(buf),
    syncModule(sync)
{
    syncs.resize(NUM_ELEMENTS, nullptr);
    postfences.resize(NUM_ELEMENTS, NvSciSyncFenceInitializer);
}

PerfClient::~PerfClient(void)
{
    if (waitContext != nullptr) {
        NvSciSyncCpuWaitContextFree(waitContext);
        waitContext = nullptr;
    }

    for (uint32_t i{ 0U }; i < packets.size(); i++) {
        for (uint32_t j{ 0U }; j < NUM_ELEMENTS; j++) {
            if (packets[i].buffers[j] != nullptr) {
                NvSciBufObjFree(packets[i].buffers[j]);
                packets[i].buffers[j] = nullptr;
            }
        }
    }

    for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
        if (syncs[i] != nullptr) {
            NvSciSyncObjFree(syncs[i]);
            syncs[i] = nullptr;
        }
    }

    if (endpointHandle != 0U) {
        NvSciStreamBlockDelete(endpointHandle);
        endpointHandle = 0U;
    }
}

void PerfClient::runPoolEventsHandler(NvSciStreamBlock pool)
{
    if (pool != 0U) {
        PoolHandler handler(pool);
        handler.handleEvents();
    } else {
        CHECK_ERR(false, "Invalid pool handle\n");
    }
}

void PerfClient::handleElemSupport(void)
{
    // Set buffer attributes for each element
    for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
        NvSciBufAttrList bufAttrLists{ nullptr };
        CHECK_NVSCIERR(NvSciBufAttrListCreate(bufModule, &bufAttrLists));
        setEndpointBufAttr(bufAttrLists);

        CHECK_NVSCIERR(
            NvSciStreamBlockElementAttrSet(endpointHandle, i, bufAttrLists));

        NvSciBufAttrListFree(bufAttrLists);
    }

    // Indicate element export done,
    //  which sends endpoint element information to the pool
    CHECK_NVSCIERR(
        NvSciStreamBlockSetupStatusSet(endpointHandle,
                                       NvSciStreamSetup_ElementExport,
                                       true));
}

void PerfClient::handleElemSetting(void)
{
    // Indicate element import done
    CHECK_NVSCIERR(
        NvSciStreamBlockSetupStatusSet(endpointHandle,
                                       NvSciStreamSetup_ElementImport,
                                       true));

    // When the packet layout is decided,
    //  set waiter attributes for each element.
    NvSciSyncAttrList waiterAttrList{ nullptr };
    if (testArg.numSyncs > 0U) {
        CHECK_NVSCIERR(NvSciSyncCpuWaitContextAlloc(syncModule, &waitContext));

        CHECK_NVSCIERR(NvSciSyncAttrListCreate(syncModule, &waiterAttrList));
        CHECK_NVSCIERR(setCpuSyncAttrList(NvSciSyncAccessPerm_WaitOnly,
                                          waiterAttrList));
    }

    for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
        CHECK_NVSCIERR(NvSciStreamBlockElementWaiterAttrSet(endpointHandle,
                                                            i,
                                                            waiterAttrList));
    }

    // Indicate waiter attribute export done,
    //  which sends waiter information to the remote endpoint
    CHECK_NVSCIERR(
        NvSciStreamBlockSetupStatusSet(endpointHandle,
                                       NvSciStreamSetup_WaiterAttrExport,
                                       true));

    NvSciSyncAttrListFree(waiterAttrList);
}

NvSciError PerfClient::setCpuSyncAttrList(NvSciSyncAccessPerm cpuPerm,
                                          NvSciSyncAttrList attrList)
{
    NvSciSyncAttrKeyValuePair keyValue[2];
    bool cpuSync{ true };
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*)&cpuSync;
    keyValue[0].len = sizeof(cpuSync);
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    return NvSciSyncAttrListSetAttrs(attrList, keyValue, 2);
}

void PerfClient::handleSyncExport(void)
{
    NvSciSyncAttrList signalerAttrList{ nullptr };

    if (testArg.numSyncs > 0U) {
        // Set signaler attributes at endpoint
        CHECK_NVSCIERR(NvSciSyncAttrListCreate(syncModule, &signalerAttrList));
        CHECK_NVSCIERR(setCpuSyncAttrList(NvSciSyncAccessPerm_SignalOnly,
                                          signalerAttrList));

        for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
            if (i < testArg.numSyncs) {
                // Retrieve waiter attributes from remote enpoint
                NvSciSyncAttrList recvSyncAttrList;
                CHECK_NVSCIERR(
                    NvSciStreamBlockElementWaiterAttrGet(endpointHandle,
                                                         i,
                                                         &recvSyncAttrList));
                if (recvSyncAttrList != nullptr) {
                    // Reconcile signaler and remote waiter attributes
                    NvSciSyncAttrList unreconciledList[2]{ signalerAttrList,
                                                           recvSyncAttrList };
                    NvSciSyncAttrList reconciledList{ nullptr };
                    NvSciSyncAttrList newConflictList{ nullptr };
                    CHECK_NVSCIERR(NvSciSyncAttrListReconcile(unreconciledList,
                                                              2U,
                                                              &reconciledList,
                                                              &newConflictList));
                    // Allocate synchronization object
                    NvSciSyncObjAlloc(reconciledList, &syncs[i]);
                    NvSciSyncAttrListFree(reconciledList);
                    NvSciSyncAttrListFree(newConflictList);
                    NvSciSyncAttrListFree(recvSyncAttrList);
                }
            }
        }

        NvSciSyncAttrListFree(signalerAttrList);
    }

    // Indicate waiter attribute import done
    CHECK_NVSCIERR(
        NvSciStreamBlockSetupStatusSet(endpointHandle,
                                       NvSciStreamSetup_WaiterAttrImport,
                                       true));

    // Set synchronization object for element
    for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
        if (syncs[i] != nullptr) {
            CHECK_NVSCIERR(NvSciStreamBlockElementSignalObjSet(endpointHandle,
                                                               i,
                                                               syncs[i]));
        }
    }

    // Indicate synchronization object export done,
    //  which sends sync objects to the remote endpoint
    CHECK_NVSCIERR(
        NvSciStreamBlockSetupStatusSet(endpointHandle,
                                       NvSciStreamSetup_SignalObjExport,
                                       true));
}

void PerfClient::handlePacketCreate(void)
{
    // Create a new packet
    Packet packet;

    // Retrieve packet handle
    CHECK_NVSCIERR(NvSciStreamBlockPacketNewHandleGet(endpointHandle,
                                                      &packet.handle));
    packet.cookie = static_cast<NvSciStreamCookie>(++numRecvPackets);
    packet.buffers.fill(nullptr);

    // Retrieve all packet buffers
    for (uint32_t i {0U}; NUM_ELEMENTS > i; ++i) {
        CHECK_NVSCIERR(NvSciStreamBlockPacketBufferGet(endpointHandle,
                                                       packet.handle,
                                                       i,
                                                       &packet.buffers[i]));
    }

    // Get the CPU pointer to read/write timestamps to buffer
    if (testArg.latency) {
        getCpuPtr(packet);
    }

    // Save the new packet
    packets.push_back(packet);

    // Send the packet status to the pool.
    CHECK_NVSCIERR(NvSciStreamBlockPacketStatusSet(endpointHandle,
                                                   packet.handle,
                                                   packet.cookie,
                                                   NvSciError_Success));
}

void PerfClient::handleSetupComplete(void)
{
    // Mark setup complete
    setupDone = true;
    // Record streaming start time
    streamingStartTime = NvpGetTimeMark();
}

void PerfClient::handleEvents(void)
{
    while (!streamingDone) {
        if (testArg.latency && setupDone) {
            // Record the packet-wait start time
            packetWaitStartTime = NvpGetTimeMark();
        }

        // Wait for new events
        NvSciStreamEventType event;
        if (!ipcNotifyOff) {
            CHECK_NVSCIERR(NvSciStreamBlockEventQuery(endpointHandle,
                                                      QUERY_WAIT_INFINITE,
                                                      &event));
        } else {
            // Busy wait to process the event immediately when it arrives.
            //
            // This test uses busy wait to avoid context switch.
            //
            // Application may make the thread sleep to reduce CPU utilization.
            // Or move this call to another thread to unblock the main thread.
            NvSciError err;
            do {
                // After disabling NvSciIpc notification, need to explicitly
                //  trigger the NvSciStream internal handler to process the
                //  remote messages.
                handleInternalEvents();

                // Check whether there's available event
                err = NvSciStreamBlockEventQuery(endpointHandle, 0U, &event);
                if (NvSciError_Timeout != err) {
                    CHECK_NVSCIERR(err);
                }
            } while (NvSciError_Success != err);
        }

        // Handle new events
        NvSciError errEvent;
        switch (event) {

        case NvSciStreamEventType_Connected:
            // Record setup start time
            setupStartTime = NvpGetTimeMark();
            handleElemSupport();
            break;

        case NvSciStreamEventType_Elements:
            handleElemSetting();
            break;

        case NvSciStreamEventType_PacketCreate:
            handlePacketCreate();
            break;

        case NvSciStreamEventType_PacketsComplete:
            // Indicate packets import done
            CHECK_NVSCIERR(
                NvSciStreamBlockSetupStatusSet(endpointHandle,
                                               NvSciStreamSetup_PacketImport,
                                               true));
            break;

        case NvSciStreamEventType_WaiterAttr:
            handleSyncExport();
            break;

        case NvSciStreamEventType_SignalObj:
            // Use CPU wait. No need to map sync object.
            //  Skip sync object query.
            // Indicate sync object import done
            CHECK_NVSCIERR(NvSciStreamBlockSetupStatusSet(
                                endpointHandle,
                                NvSciStreamSetup_SignalObjImport,
                                true));
            break;

        case NvSciStreamEventType_SetupComplete:
            handleSetupComplete();
            break;

        case NvSciStreamEventType_PacketReady:
            handlePayload();
            break;

        case NvSciStreamEventType_Disconnected:
            handleStreamComplete();
            return;

        case NvSciStreamEventType_Error:
            // Query error code of the error event
            CHECK_NVSCIERR(NvSciStreamBlockErrorGet(endpointHandle,
                                                    &errEvent));
            printf("ERR: Received error event %x\n", errEvent);
            return;

        default:
            CHECK_ERR(false, "Received unexpected event");
        }
    }
}
