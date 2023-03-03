/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CNVMTRANSPORTINK_MAX96712_96705_HPP_
#define _CNVMTRANSPORTINK_MAX96712_96705_HPP_

#include "TransportLinkIF/CNvMTransportLink.hpp"
#include "utils/utils.hpp"

namespace nvsipl
{

class CNvMTransportLink_Max96712_96705 final: public CNvMTransportLink
{
public:
    virtual SIPLStatus Init(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg) override;

    virtual SIPLStatus PostSensorInit(
        DevBlkCDIDevice const* const brdcstSerCDI,
         uint8_t const linkMask, bool const groupInitProg) const override;

    virtual SIPLStatus MiscInit() const override;

    virtual SIPLStatus Reset() const override;

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

    virtual bool IsGMSL2() const override
    {
        return false;
    };

private:
#if !NV_IS_SAFETY
    // Dump link Parameters
    void DumpLinkParams();
#endif

    // Setup config link
    SIPLStatus SetupConfigLink(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg);

    // Setup serializer address translations
    SIPLStatus SetupAddressTranslationsSer(DevBlkCDIDevice* brdcstSerCDI, bool groupInitProg);

    // Setup eeprom address translations
    SIPLStatus SetupAddressTranslationsEEPROM() const;

    // Setup video link
    SIPLStatus SetupVideoLink();
};

} // end of namespace nvsipl
#endif // _CNVMTRANSPORTINK_MAX96712_96705_HPP_
