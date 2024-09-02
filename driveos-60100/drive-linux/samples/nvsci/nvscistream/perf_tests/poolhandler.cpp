//! \file
//! \brief NvSciStream test Pool Handler declaration.
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

#include <unistd.h>
#include "poolhandler.h"

extern TestArg testArg;

PoolHandler::PoolHandler(NvSciStreamBlock poolHandle):
    pool(poolHandle)
{
    // Consumer process in c2c stream uses a c2c pool.
    isC2cPool = testArg.isC2c && (testArg.testType == CrossProcCons);

    allocatedPackets.resize(testArg.numPackets);
    for (uint32_t i{ 0U }; i < testArg.numPackets; i++) {
        allocatedPackets[i].buffers.fill(nullptr);
    }
}

PoolHandler::~PoolHandler(void)
{
    for (uint32_t i{ 0U }; i < allocatedPackets.size(); i++) {
        for (uint32_t j{ 0U }; j < NUM_ELEMENTS; j++) {
            if (allocatedPackets[i].buffers[j] != nullptr) {
                NvSciBufObjFree(allocatedPackets[i].buffers[j]);
            }
        }
    }
}

void PoolHandler::handleEvents(void)
{
    while (true) {
        // Wait for new events
        NvSciStreamEventType event;
        CHECK_NVSCIERR(NvSciStreamBlockEventQuery(pool,
                                                  QUERY_WAIT_INFINITE,
                                                  &event));
        // Handle new events
        NvSciError errEvent;
        switch (event) {

        case NvSciStreamEventType_Connected:
            // Does nothing
            break;

        case NvSciStreamEventType_Elements:
            handleBufferSetup();
            break;

        case NvSciStreamEventType_PacketStatus:
            // Both endpoints set success for packet status in this test,
            //  skip packet status query.
            // Indicate packet import done after receiving all packet stauts.
            if (++numPacketStatus == testArg.numPackets) {
                CHECK_NVSCIERR(
                    NvSciStreamBlockSetupStatusSet(pool,
                                                   NvSciStreamSetup_PacketImport,
                                                   true));
            }
            break;

        case NvSciStreamEventType_SetupComplete:
            // Does nothing
            break;

        case NvSciStreamEventType_Disconnected:
            return;

        case NvSciStreamEventType_Error:
            // Query error code of the error event on pool
            CHECK_NVSCIERR(NvSciStreamBlockErrorGet(pool, &errEvent));
            printf("ERR: Pool received error event %x\n", errEvent);
            return;

        default:
            CHECK_ERR(false, "Pool received unexpected event");
        };
    }
}

void PoolHandler::handleBufferSetup(void)
{
    std::array<NvSciBufAttrList, NUM_ELEMENTS> allocatedAttrs;

    // Primary pool queries element information from endpoints,
    //  and reconciles the attributes.
    if (!isC2cPool) {
        uint32_t numProdAttr{ 0U };
        uint32_t numConsAttr{ 0U };
        std::array<NvSciBufAttrList, NUM_ELEMENTS> prodAttrs;
        std::array<NvSciBufAttrList, NUM_ELEMENTS> consAttrs;

        // Query element count
        CHECK_NVSCIERR(
            NvSciStreamBlockElementCountGet(pool,
                                            NvSciStreamBlockType_Producer,
                                            &numProdAttr));
        CHECK_ERR(NUM_ELEMENTS == numProdAttr,
                  "Incorrect element count from producer");

        CHECK_NVSCIERR(
            NvSciStreamBlockElementCountGet(pool,
                                            NvSciStreamBlockType_Consumer,
                                            &numConsAttr));
        CHECK_ERR(NUM_ELEMENTS == numConsAttr,
                  "Incorrect element count from consumer");

        // Query buffer attributes
        for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
            // This test has only one element in packet and
            //  sets with the same type. Skip element type query.
            CHECK_NVSCIERR(
                NvSciStreamBlockElementAttrGet(pool,
                                               NvSciStreamBlockType_Producer,
                                               i,
                                               nullptr,
                                               &prodAttrs[i]));

            CHECK_NVSCIERR(
                NvSciStreamBlockElementAttrGet(pool,
                                               NvSciStreamBlockType_Consumer,
                                               i,
                                               nullptr,
                                               &consAttrs[i]));
        }

        for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
            // Reconcile buffer attributes for each element from endpoints
            NvSciBufAttrList endpointAttr[2] = { prodAttrs[i], consAttrs[i] };
            NvSciBufAttrList conflictlist{ nullptr };
            CHECK_NVSCIERR(NvSciBufAttrListReconcile(endpointAttr,
                                                     2U,
                                                     &allocatedAttrs[i],
                                                     &conflictlist));

            // Set reconciled attributes
            CHECK_NVSCIERR(NvSciStreamBlockElementAttrSet(pool,
                                                          i,
                                                          allocatedAttrs[i]));

            NvSciBufAttrListFree(prodAttrs[i]);
            NvSciBufAttrListFree(consAttrs[i]);
        }
    }

    // C2C pool queries reconciled element information from primary pool.
    else {
        // Query element count
        uint32_t numAttr{ 0U };
        CHECK_NVSCIERR(
            NvSciStreamBlockElementCountGet(pool,
                                            NvSciStreamBlockType_Producer,
                                            &numAttr));
        CHECK_ERR(NUM_ELEMENTS == numAttr,
                 "Incorrect element count from primary pool");

        // Query buffer attributes for each element
        for (uint32_t i{ 0U }; i < numAttr; i++) {
            // This test has only one element in packet and
            //  sets with the same type. Skip element type query.
            CHECK_NVSCIERR(
                NvSciStreamBlockElementAttrGet(pool,
                                               NvSciStreamBlockType_Producer,
                                               i,
                                               nullptr,
                                               &allocatedAttrs[i]));
        }
    }

    // Indicate element import done
    CHECK_NVSCIERR(
        NvSciStreamBlockSetupStatusSet(pool,
                                       NvSciStreamSetup_ElementImport,
                                       true));

    // Indicate element specification done,
    //  which sends reconciled element information to endpoints.
    if (!isC2cPool) {
        CHECK_NVSCIERR(
            NvSciStreamBlockSetupStatusSet(pool,
                                           NvSciStreamSetup_ElementExport,
                                           true));
    }


    // Create packet and buffer objects
    for (uint32_t i{ 0U }; i < testArg.numPackets; i++) {
        Packet *packet = &allocatedPackets[i];
        packet->cookie = static_cast<NvSciStreamCookie>(i + 1U);

        // Create new packet
        CHECK_NVSCIERR(NvSciStreamPoolPacketCreate(pool,
                                                   packet->cookie,
                                                   &packet->handle));

        // Allocate buffer for each element in packet
        for (uint32_t j{ 0U }; j < NUM_ELEMENTS; j++) {
            CHECK_NVSCIERR(NvSciBufObjAlloc(allocatedAttrs[j],
                                            &packet->buffers[j]));

            CHECK_NVSCIERR(
                NvSciStreamPoolPacketInsertBuffer(pool,
                                                  packet->handle,
                                                  j,
                                                  packet->buffers[j]));
        }
        // Indicate packet create complete,
        //  which sends the new packet to endpoints
        CHECK_NVSCIERR(NvSciStreamPoolPacketComplete(pool, packet->handle));
    }

    // Indicate packets export done
    CHECK_NVSCIERR(
        NvSciStreamBlockSetupStatusSet(pool,
                                       NvSciStreamSetup_PacketExport,
                                       true));

    for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
        NvSciBufAttrListFree(allocatedAttrs[i]);
    }
}
