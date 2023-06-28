/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef CWFDCONSUMER_HPP
#define CWFDCONSUMER_HPP

using namespace std;
using namespace nvsipl;

const static uint32_t RECON_NUM = 3U;

class CWFDConsumer
{
public:
    CWFDConsumer();

    virtual ~CWFDConsumer();

    SIPLStatus Init(NvSciBufModule &bufModule,
                    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attrList);
    NvSciBufObj & GetBuffer(void);
    SIPLStatus WfdFlip(void);

private:
    class CNvSciBufResources
    {
    public:

        CNvSciBufResources();
        ~CNvSciBufResources();
        SIPLStatus InitBufRes(NvSciBufModule& bufModule,
                              NvSciBufObj &bufObj,
                              std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attrList);

    private:

        NvSciBufModule m_bufModule;
        NvSciBufAttrList m_unreconciledList[RECON_NUM];
        NvSciBufAttrList m_conflictList;
        NvSciBufAttrList m_reconciledList;
    };

    SIPLStatus InitWFD();

    WFDDevice m_wfdDevice;
    WFDPort m_wfdPort;
    WFDPipeline m_wfdPipeline;
    WFDSource m_wfdSource;
    CNvSciBufResources m_nvSciBufRes;
    NvSciBufObj m_bufObj;

    bool m_inited;
};

#endif //CWFDCONSUMER_HPP
