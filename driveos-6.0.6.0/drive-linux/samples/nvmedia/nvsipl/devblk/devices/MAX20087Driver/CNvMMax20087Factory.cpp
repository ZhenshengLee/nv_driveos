/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMMax20087Factory.hpp"
#include "CNvMMax20087SyncAdapter.hpp"
#include "sipl_error.h"

namespace nvsipl
{
std::map<CNvMMax20087DriverFactory::DriverKey,
         Impl::CNvMMax20087SyncAdapter,
         CNvMMax20087DriverFactory::DriverMapCmpByI2CBus>
  CNvMMax20087DriverFactory::ms_BusToDriverMap{};

std::set<CNvMMax20087DriverFactory::DriverKey,
         CNvMMax20087DriverFactory::DriverMapCmpByLexo>
  CNvMMax20087DriverFactory::ms_BusToLinkSet{};

std::mutex CNvMMax20087DriverFactory::ms_lock{};

std::unique_ptr<CNvMCampwr>
CNvMMax20087DriverFactory::RequestPowerDriver(DevBlkCDIRootDevice *cdiRootDev, uint8_t linkIdx)
{
    std::lock_guard<std::mutex> lock(ms_lock);

    auto set_entry = ms_BusToLinkSet.insert({cdiRootDev, linkIdx});
    if (!set_entry.second) {
        SIPL_LOG_ERR_STR("CNvMMax20087 Driver Factory detected duplicate illegal access");
        return nullptr;
    }

    Impl::CNvMMax20087SyncAdapter* adapter = [&] {
        const DriverKey key{cdiRootDev, linkIdx};

        auto search = ms_BusToDriverMap.find(key);
        if (search == ms_BusToDriverMap.end()) {
            LOG_DEBUG("CNvMMax20087 Driver Factory creating new adapter for CDI root dev %p\n",
                      cdiRootDev);

            // This is a new driver, insert it with its default constructor
            auto entry = ms_BusToDriverMap.emplace(std::piecewise_construct,
                                                   std::forward_as_tuple(key),
                                                   std::tuple<>{});
            return &entry.first->second;
        } else {
            LOG_DEBUG("CNvMMax20087 Driver Factory found an existing driver for CDI root dev %p\n",
                      cdiRootDev);

            // The driver already exits
            return &search->second;
        }
    }();

    return std::make_unique<Impl::CNvMMax20087AccessToken>(adapter, linkIdx);
}

} // end of namespace nvsipl