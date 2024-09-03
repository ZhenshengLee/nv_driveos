/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _CNVMTRANSPORTINK_MAX96712_TPGSERIALIZER_HPP_
#define _CNVMTRANSPORTINK_MAX96712_TPGSERIALIZER_HPP_

#include "TransportLinkIF/CNvMTransportLink.hpp"

namespace nvsipl
{

class CNvMTransportLink_Max96712_TPGSerializer: public CNvMTransportLink
{
public:
    virtual SIPLStatus Init(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg) override;

    virtual SIPLStatus PostSensorInit(uint8_t const linkMask, bool const groupInitProg) const override
    {
        return NVSIPL_STATUS_OK;
    }

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
#if !NV_IS_SAFETY
    // Dump link Parameters
    void DumpLinkParams();
#endif

};

} // end of namespace

#endif /* _CNVMTRANSPORTINK_MAX96712_TPGSERIALIZER_HPP_ */
