/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CNVMDS90UB9724_HPP_
#define _CNVMDS90UB9724_HPP_

#include "DeserializerIF/CNvMDeserializer.hpp"
#include "utils/utils.hpp"

namespace nvsipl
{

extern "C" {
#include "cdi_ds90ub9724.h"
}

//! Class for DS90UB9724 deserialzer
class CNvMDS90UB9724 final: public CNvMDeserializer
{
public:
    ~CNvMDS90UB9724();

    SIPLStatus DoSetConfig(const DeserInfo *deserInfo, DeserializerParams *params) override;

    SIPLStatus DoSetPower(bool powerOn) override final;

    SIPLStatus DoInit() override;

    SIPLStatus EnableLinks(uint8_t linkMask) override;

    SIPLStatus ControlLinks(const std::vector<LinkAction>& linkActions) override;

    SIPLStatus CheckLinkLock(uint8_t linkMask) override;

    SIPLStatus DoStart() override;

    SIPLStatus GetErrorSize(size_t & errorSize) override;

    SIPLStatus GetErrorInfo(std::uint8_t * const buffer,
                            const std::size_t bufferSize,
                            std::size_t &size,
                            bool & isRemoteError,
                            std::uint8_t& linkErrorMask) override;

    Interface* GetInterface(const UUID &interfaceId) override;

    virtual SIPLStatus GetInterruptStatus(
        const uint32_t gpioIdx,
        IInterruptNotify &intrNotifier) override;

private:
    SIPLStatus SetDS90UB9724Ctx();

    //! Holds the revision of the DS90UB9724
    RevisionDS90UB9724 m_eRevision;
};

} // end of namespace nvsipl

#endif // _CNVMDS90UB9724_HPP_
