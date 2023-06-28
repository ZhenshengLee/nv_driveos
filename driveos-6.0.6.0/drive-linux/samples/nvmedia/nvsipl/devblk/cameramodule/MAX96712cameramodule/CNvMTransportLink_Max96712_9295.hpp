/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CNVMTRANSPORTINK_MAX96712_9295_HPP_
#define _CNVMTRANSPORTINK_MAX96712_9295_HPP_

#include "TransportLinkIF/CNvMTransportLink.hpp"

namespace nvsipl
{

class CNvMTransportLink_Max96712_9295: public CNvMTransportLink
{
public:
    virtual SIPLStatus Init(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg) override;

    virtual SIPLStatus PostSensorInit(uint8_t const linkMask, bool const groupInitProg) const override;

    virtual SIPLStatus MiscInit() const override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus Start() const override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus Deinit() const
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus Stop() const override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus Reset() const override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual bool IsGMSL2() const override
    {
        return true;
    };

private:
#if !NV_IS_SAFETY
    // Dump link Parameters
    void DumpLinkParams();
#endif

    // Setup address translations
    SIPLStatus SetupAddressTranslations(DevBlkCDIDevice* brdcstSerCDI);
};

} // end of namespace nvsipl
#endif // _CNVMTRANSPORTINK_MAX96712_9295_HPP_
