/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#ifndef __CNVMTRANSPORTINK_MAX96712_96717_HPP__
#define __CNVMTRANSPORTINK_MAX96712_96717_HPP__
#include "TransportLinkIF/CNvMTransportLink.hpp"
namespace nvsipl
{
class CNvMTransportLink_Max96712_96717: public CNvMTransportLink
{
public:
    virtual SIPLStatus Init(DevBlkCDIDevice* brdcstSerISC, uint8_t linkMask, bool groupInitProg) override;

    virtual SIPLStatus PostSensorInit(DevBlkCDIDevice const* const brdcstSerISC, uint8_t const linkMask, bool const groupInitProg) const override;

    virtual SIPLStatus MiscInit() const override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus Start() const override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus Deinit() const override
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
    // Dump link Parameters
    void DumpLinkParams();
    // Setup address translations
    SIPLStatus SetupAddressTranslations(DevBlkCDIDevice* brdcstSerISC);
};
} // end of namespace nvsipl
#endif // __CNVMTRANSPORTINK_MAX96712_96717_HPP__
