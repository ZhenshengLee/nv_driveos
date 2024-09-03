//! \file
//! \brief NvSciStream test Pool Handler declaration.
//!
//! \copyright
//! Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//!
//! NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
//! property and proprietary rights in and to this material, related
//! documentation and any modifications thereto. Any use, reproduction,
//! disclosure or distribution of this material and related documentation
//! without an express license agreement from NVIDIA CORPORATION or
//! its affiliates is strictly prohibited.

#ifndef POOLHANDLER_H
#define POOLHANDLER_H

#include "util.h"
#include <vector>

class PoolHandler
{
public:
    PoolHandler(NvSciStreamBlock poolHandle);
    ~PoolHandler(void);

    // Handle all events on pool
    //  with the following handle*() helper functions.
    void handleEvents(void);

private:
    // After receiving the element information,
    //  the primary pool reconciles element attributes from producer and
    //  consumer, and provides the reconciled attributes back to the endpoints;
    //  the secondary (c2c) pool retrieves the reconciled attributes from
    //  the primary pool.
    //  Both types of pool create new packets and allocate buffer objects with
    //  the reconciled attriubtes.
    void handleBufferSetup(void);

private:
    NvSciStreamBlock       pool{ 0U };
    bool                   isC2cPool{ false };

    std::vector<Packet>    allocatedPackets;
    uint32_t               numPacketStatus{ 0U };
} ;

#endif // POOLHANDLER_H
