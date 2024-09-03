/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CNVMTRANSPORTINK_DS90UB9724_971_HPP_
#define _CNVMTRANSPORTINK_DS90UB9724_971_HPP_

#include "TransportLinkIF/CNvMTransportLink.hpp"

namespace nvsipl
{

class CNvMTransportLink_DS90UB9724_971: public CNvMTransportLink
{
public:
    SIPLStatus Init(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg) override;

    // Setup config link
    SIPLStatus SetupConfigLink(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg);

    virtual SIPLStatus MiscInit() const override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus PostSensorInit(uint8_t const linkMask, bool const groupInitProg) const
    {
        return NVSIPL_STATUS_OK;
    };

    bool IsGMSL2() const override {return false;};

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

private:
#if !NV_IS_SAFETY
    // Dump link Parameters
    void DumpLinkParams();
#endif

    // Setup address translations
    SIPLStatus SetupAddressTranslations(DevBlkCDIDevice* brdcstSerCDI);
};

} // end of namespace nvsipl
#endif // _CNVMTRANSPORTINK_DS90UB9724_971_HPP_
