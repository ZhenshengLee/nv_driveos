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
static const uint32_t BUFFER_NUM = 6U;
static const uint32_t BUFFER_NUM_SOLO = 2U;

class CNvWfdResources final : public IDisplayInterface
{
public:
    CNvWfdResources() = delete;
    CNvWfdResources(WFDDevice wfdDevice, WFDint wfdPortId);
    ~CNvWfdResources();
    SIPLStatus Init(uint32_t &uWidth,
                    uint32_t &uHeight,
                    NvSciBufModule &bufModule,
                    NvSciBufAttrList &bufAttrList,
                    bool soloInput) override;
    void GetBuffers(std::vector<NvSciBufObj>& vBuffers) override;
    SIPLStatus WfdClearDisplay() override;
    SIPLStatus WfdFlip(NvSciBufObj& bufObj) override;
private:
    class CNvSciBufResources
    {
    public:
        CNvSciBufResources();
        ~CNvSciBufResources();
        SIPLStatus InitBufRes(NvSciBufModule &bufModule,
                              std::vector<NvSciBufObj> &vBufObj,
                              NvSciBufAttrList &bufAttrList,
                              uint32_t uWidth,
                              uint32_t uHeight,
                              bool soloInput);
    private:
        NvSciBufAttrList m_unreconciledList[RECON_NUM];
        NvSciBufAttrList m_conflictList;
        NvSciBufAttrList m_reconciledList;
    };

    SIPLStatus InitWFD();

    WFDDevice m_wfdDevice;
    WFDint m_wfdPortId = WFD_INVALID_PORT_ID;
    WFDPort m_wfdPort = WFD_INVALID_HANDLE;
    WFDPipeline m_wfdPipeline[2] = {WFD_INVALID_HANDLE};
    std::vector<WFDint> m_wfdBindablePipeIds;
    CNvSciBufResources m_nvSciBufRes;
    bool m_inited = false;
    bool m_soloInput;
    std::vector<std::pair<NvSciBufObj,std::pair<WFDPipeline, WFDSource>>> m_vBufSourcePair;
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
