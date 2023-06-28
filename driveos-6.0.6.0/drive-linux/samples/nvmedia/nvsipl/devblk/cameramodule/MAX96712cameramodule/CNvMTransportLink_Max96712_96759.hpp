/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CNVMTRANSPORTINK_MAX96712_96759_HPP_
#define _CNVMTRANSPORTINK_MAX96712_96759_HPP_

#include "TransportLinkIF/CNvMTransportLink.hpp"
#include "utils/utils.hpp"

namespace nvsipl
{

class CNvMTransportLink_Max96712_96759 final: public CNvMTransportLink
{
public:
    virtual SIPLStatus Init(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg) override;

    virtual bool IsGMSL2() const override
    {
        return true;
    };

    virtual SIPLStatus PostSensorInit(uint8_t const linkMask, bool const groupInitProg) const
    {
        return NVSIPL_STATUS_OK;
    };

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


private:
    // Dump link Parameters
    void DumpLinkParams();

    // Setup address translations
    SIPLStatus SetupAddressTranslations(DevBlkCDIDevice* brdcstSerCDI);

};

} // end of namespace nvsipl
#endif // _CNVMTRANSPORTINK_MAX96712_96759_HPP_
