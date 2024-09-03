/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CNVMMAX96712_HPP_
#define _CNVMMAX96712_HPP_

#include "DeserializerIF/CNvMDeserializer.hpp"
#include "utils/utils.hpp"

namespace nvsipl
{

extern "C" {
#include "cdi_max96712.h"
}

//! Class for Max96712 deserialzer
class CNvMMax96712 final: public CNvMDeserializer
{
public:
    ~CNvMMax96712();

    SIPLStatus EnableLinks(uint8_t linkMask) override;

    SIPLStatus ControlLinks(const std::vector<LinkAction>& linkActions) override;

    SIPLStatus CheckLinkLock(uint8_t linkMask) override;

    SIPLStatus GetErrorSize(size_t & errorSize) override;

    SIPLStatus GetErrorInfo(std::uint8_t * const buffer, const std::size_t bufferSize,
                            std::size_t &size, bool & isRemoteError,
                            std::uint8_t& linkErrorMask) override;

    SIPLStatus DoSetPower(bool powerOn) override final;
    Interface* GetInterface(const UUID &interfaceId) override;

    virtual SIPLStatus GetInterruptStatus(
        const uint32_t gpioIdx,
        IInterruptNotify &intrNotifier) override;

private:
    SIPLStatus SetMAX96712Ctx();

    virtual SIPLStatus DoSetConfig(const DeserInfo *deserInfo, DeserializerParams *params) override;

    virtual SIPLStatus DoInit() override;

    virtual SIPLStatus DoStart() override;

    virtual SIPLStatus DoStop() override
    {
        return NVSIPL_STATUS_OK;
    };

    //! Holds the revision of the MAX96712
    RevisionMAX96712 m_eRevision;

    /*! Name of deserializer */
    std::string m_sDeserializerName;
};

} // end of namespace nvsipl
#endif //_CNVMMAX96712_HPP_
