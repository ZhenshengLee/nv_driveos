/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMMAX20087_FACTORY_HPP
#define CNVMMAX20087_FACTORY_HPP

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>

#include "CNvMMax20087SyncAdapter.hpp"
#include "CampwrIF/CNvMCampwr.hpp"
#include "devblk_cdi.h"

namespace nvsipl
{
/**
 * @brief The driver factory that manages the lifetime of the power load
 * swtich drivers.
 *
 * An internal set manages access control such that no unique combination
 * of a root device & link index can be given out twice.
 */
class CNvMMax20087DriverFactory
{
  public:
    /**
     * \brief  Requests for a new power control driver.
     *
     * @param[in] cdiRootDev  Pointer to a CDI root device.
     * @param[in] linkIndex   The data link index;
     *                        Valid range: [0, (Maximum Links Supported per Deserializer - 1)].
     * @retval                The new power load switch driver's handle or NULL if error occurred.
     */
    static std::unique_ptr<CNvMCampwr> RequestPowerDriver(DevBlkCDIRootDevice *cdiRootDev,
                                                          uint8_t linkIdx);

  private:
    CNvMMax20087DriverFactory() = default;

    struct DriverKey
    {
        DevBlkCDIRootDevice *cdiRootDev;
        uint8_t linkIdx;
    };

    struct DriverMapCmpByI2CBus
    {
        bool operator()(const DriverKey& lhs,
                        const DriverKey& rhs) const noexcept
        {
            return lhs.cdiRootDev < rhs.cdiRootDev;
        }
    };

    struct DriverMapCmpByLexo
    {
        bool operator()(const DriverKey& lhs,
                        const DriverKey& rhs) const noexcept
        {
            return std::tie(lhs.cdiRootDev, lhs.linkIdx) <
                   std::tie(rhs.cdiRootDev, rhs.linkIdx);
        }
    };

    /**
     * \brief Association of driver keys to sync adapter objects where each
     * unique root device maps to a unique sync adapter object.
     */
    static std::map<DriverKey, Impl::CNvMMax20087SyncAdapter, DriverMapCmpByI2CBus>
      ms_BusToDriverMap;

    /**
     * \brief Tracking set to ensure any unique combination of a root device
     * and link index can only be requested once.
     */
    static std::set<DriverKey, DriverMapCmpByLexo> ms_BusToLinkSet;

    static std::mutex ms_lock;
};

} // end of namespace nvsipl
#endif // CNVMMAX20087_FACTORY_HPP