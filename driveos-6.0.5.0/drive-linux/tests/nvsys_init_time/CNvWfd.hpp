/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>

// The extension string for WFD_NVX_create_source_from_nvscibuf
// need to be defined before including the WFD headers
#define WFD_NVX_create_source_from_nvscibuf
#define WFD_WFDEXT_PROTOTYPES
#include <WF/wfd.h>
#include <WF/wfdext.h>

#include "NvSIPLClient.hpp"
#include "nvmedia_2d.h"
#include "CUtils.hpp"

#ifndef CNVWFD_HPP
#define CNVWFD_HPP

using namespace nvsipl;

static const uint32_t RECON_NUM = 3U;

class CNvWfdResources final : public IDisplayInterface
{
public:
    CNvWfdResources() = delete;
    CNvWfdResources(WFDDevice wfdDevice, WFDint wfdPortId);
    ~CNvWfdResources();
    SIPLStatus Init(uint32_t &uWidth,
                    uint32_t &uHeight,
                    NvSciBufModule &bufModule,
                    NvSciBufAttrList &bufAttrList) override;
    NvSciBufObj & GetBuffer() override;
    SIPLStatus WfdFlip() override;
private:
    class CNvSciBufResources
    {
    public:
        CNvSciBufResources();
        ~CNvSciBufResources();
        SIPLStatus InitBufRes(NvSciBufModule &bufModule,
                              NvSciBufObj &bufObj,
                              NvSciBufAttrList &bufAttrList,
                              uint32_t uWidth,
                              uint32_t uHeight);
    private:
        NvSciBufAttrList m_unreconciledList[RECON_NUM];
        NvSciBufAttrList m_conflictList;
        NvSciBufAttrList m_reconciledList;
    };

    SIPLStatus InitWFD();

    WFDDevice m_wfdDevice;
    WFDint m_wfdPortId = WFD_INVALID_PORT_ID;
    WFDPort m_wfdPort = WFD_INVALID_HANDLE;
    WFDPipeline m_wfdPipeline = WFD_INVALID_HANDLE;
    WFDSource m_wfdSource = WFD_INVALID_HANDLE;
    CNvSciBufResources m_nvSciBufRes;
    NvSciBufObj m_bufObj = nullptr;
    bool m_inited = false;
};

class CNvWfdResourcesCommon final : public IDisplayManager
{
public:
    CNvWfdResourcesCommon() = default;
    ~CNvWfdResourcesCommon();
    SIPLStatus Init(uint32_t uNumDisplays) override;
    SIPLStatus GetDisplayInterface(uint32_t uDispId, IDisplayInterface * &pDispIf) override;
private:
    WFDDevice m_wfdDevice;
    WFDint m_wfdPortIds[MAX_SUPPORTED_DISPLAYS] = {WFD_INVALID_PORT_ID};
    CNvWfdResources *m_nvWfd[MAX_SUPPORTED_DISPLAYS] = {nullptr};
};

#endif // CNVWFD_HPP
