/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CPLUGIN_HPP
#define CPLUGIN_HPP

/* STL Headers */
#include <cstring>

#include "INvSiplControlAuto.hpp"

#define ISP_OUT_MAX_BITS (20U)
#define ABS(x)  ((x) < 0 ? -(x) : (x))
#define CLIP(x, min, max)  ((x) > (max) ? (max) : ((x) < (min) ? (min) : (x)))

class CAutoControlPlugin : public nvsipl::ISiplControlAuto
{

public:
    CAutoControlPlugin();
    ~CAutoControlPlugin() = default;

    nvsipl::SIPLStatus Process(const nvsipl::SiplControlAutoInputParam& inParam,
                               nvsipl::SiplControlAutoOutputParam& outParam) noexcept override;

    nvsipl::SIPLStatus Reset();

private:
    CAutoControlPlugin(const CAutoControlPlugin&) = delete;
    CAutoControlPlugin& operator=(const CAutoControlPlugin&) = delete;
    nvsipl::SIPLStatus ProcessAE(const nvsipl::SiplControlAutoInputParam& inParam,
                                 nvsipl::SiplControlAutoSensorSetting& sensorSett);


    nvsipl::SIPLStatus ProcessAWB(const nvsipl::SiplControlAutoInputParam& inParam,
                                  nvsipl::SiplControlAutoOutputParam& outParam);


    // Sample plug-in
    float_t m_prevExpVal;
    float_t m_targetLuma;
    float_t m_dampingFactor;
};

//factories for dynamic loading
extern "C" nvsipl::ISiplControlAuto* CreatePlugin();


#endif // CPLUGIN_HPP
