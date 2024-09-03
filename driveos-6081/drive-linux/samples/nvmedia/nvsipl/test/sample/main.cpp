/*
 * Copyright (c) 2020 - 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Standard header files
#include <iostream>
#include <memory>
#include <thread>
#include <mutex>

#if !NVMEDIA_QNX
#include <fstream>
#else // NVMEDIA_QNX
#include "nvdtcommon.h"
#endif // !NVMEDIA_QNX

// SIPL header files
#include "NvSIPLCommon.hpp"
#include "NvSIPLCamera.hpp"
#include "NvSIPLClient.hpp"
#include "NvSIPLPipelineMgr.hpp"

// Sample application header files
#include "CUtils.hpp"
#include "CImageManager.hpp"
#include "QueueHandlers.hpp"
#ifdef NVMEDIA_QNX
#include "CComposite.hpp"
#endif // NVMEDIA_QNX

// Platfom configuration header files
#include "platform/common.hpp"
#include "platform/imx390_c.hpp"
#include "platform/ar0820_b.hpp"
#include "platform/ov2311_c.hpp"
#include "platform/vc0820_b.hpp"
#include "platform/ov2311.hpp"
#include "platform/imx728vb2.hpp"
#include "platform/imx623vb2.hpp"
#include "platform/ar0820.hpp"

#ifndef NITO_PATH
    #define NITO_PATH "/usr/share/camera/"
#endif

using namespace nvsipl;

/**
 * Control information for a single queue.
 * Only one of @c imageQueueHandler and @c eventQueueHandler will be active,
 * depending on the type of queue being monitored.
 */
struct ThreadData: public FrameCompleteQueueHandler::ICallback,
                   public NotificationQueueHandler::ICallback
{
    std::string threadName;
    FrameCompleteQueueHandler imageQueueHandler;
    NotificationQueueHandler eventQueueHandler;
    std::mutex *printMutex;
    uint32_t uSensorId = MAX_NUM_SENSORS;
    uint32_t uNumFrameDrops = 0U;
    uint32_t uNumFrameDiscontinuities = 0U;
#ifdef NVMEDIA_QNX
    CComposite *pComposite = nullptr;
#endif // NVMEDIA_QNX

    /**
     * Process events from the notification queue (if applicable).
     */
    void process(const NvSIPLPipelineNotifier::NotificationData& event)
    {
        if (event.eNotifType < NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_FRAME_DROP) {
            // Don't print information events, these completion notifications are redundant since
            // the frame sequence counter is already being printed
            return;
        } else if (event.eNotifType == NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_FRAME_DROP) {
            uNumFrameDrops++;
        } else if (event.eNotifType == NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_FRAME_DISCONTINUITY) {
            uNumFrameDiscontinuities++;
        }

        const char *eventName = nullptr;
        const SIPLStatus status = GetEventName(event, eventName);
        if ((status != NVSIPL_STATUS_OK) || (eventName == nullptr)) {
            LOG_ERR("Failed to get event name\n");
            return;
        } else {
            printMutex->lock();
            std::cout << threadName << ": " << eventName << std::endl;
            printMutex->unlock();
        }
    }

    /**
     * Process completed images (if applicable).
     */
    void process(INvSIPLClient::INvSIPLBuffer* const & pBuffer)
    {
        INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer =
            static_cast<INvSIPLClient::INvSIPLNvMBuffer *>(pBuffer);
        if (pNvMBuffer == nullptr) {
            LOG_ERR("Invalid buffer\n");
        } else {
            if (uSensorId >= MAX_NUM_SENSORS) {
                LOG_ERR("%s: Invalid sensor index\n", __func__);
                std::terminate();
            }
            const INvSIPLClient::ImageMetaData& metadata = pNvMBuffer->GetImageData();
            if (!metadata.frameSeqNumInfo.frameSeqNumValid) {
                LOG_ERR("Invalid frame sequence number\n");
            } else {
                printMutex->lock();
                std::cout << threadName << ": " \
                          << metadata.frameSeqNumInfo.frameSequenceNumber << std::endl;
                printMutex->unlock();
            }
#ifdef NVMEDIA_QNX
            if (pComposite != nullptr) {
                SIPLStatus status = pComposite->Post(uSensorId, pNvMBuffer);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("CComposite::Post failed\n");
                }
            }
#endif // NVMEDIA_QNX
            const SIPLStatus status = pNvMBuffer->Release();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Buffer release failed\n");
            }
        }
    }
};

enum ThreadIndex {
    THREAD_INDEX_ICP = 0U,
    THREAD_INDEX_ISP0,
    THREAD_INDEX_ISP1,
    THREAD_INDEX_ISP2,
    THREAD_INDEX_EVENT,
    THREAD_INDEX_COUNT
};

static void PrintUsage()
{
    LOG_MSG("Usage: nvsipl_sample [options]\n");
    LOG_MSG("Options:\n");
    LOG_MSG("-h                   : Print usage\n");
    LOG_MSG("-p <platformCfgName> : Specify platform configuration, default is LI-OV2311-VCSEL-GMSL2-60H_DPHY_x4\n");
    LOG_MSG("-v <level>           : Set verbosity\n");
    LOG_MSG("                     :   Supported values (default: %d):\n", LEVEL_ERR);
    LOG_MSG("                     :     %d (None)\n", LEVEL_NONE);
    LOG_MSG("                     :     %d (Errors)\n", LEVEL_ERR);
    LOG_MSG("                     :     %d (Warnings and above)\n", LEVEL_WARN);
    LOG_MSG("                     :     %d (Info and above)\n", LEVEL_INFO);
    LOG_MSG("                     :     %d (Debug and above)\n", LEVEL_DBG);
#ifdef NVMEDIA_QNX
    LOG_MSG("-d                   : Enable display\n");
#endif // NVMEDIA_QNX
}

static SIPLStatus ParseArgs(int argc, char *argv[], char *platformCfgName, bool* bEnableDisp)
{
    int i = 1;
    bool bLastArg = false;
    bool bDataAvailable = false;

    while (i < argc && argv[i][0] == '-') {
        // Check if this is the last argument
        bLastArg = ((argc - i) == 1);

        // Check if there is data available to be parsed following the option
        bDataAvailable = (!bLastArg) && !(argv[i+1][0] == '-');

        if (!strcmp(argv[i], "-h")) {
            PrintUsage();
            // Return a unique code to signal that application should exit
            return NVSIPL_STATUS_EOF;
        } else if (!strcmp(argv[i], "-p")) {
            if (bDataAvailable) {
                if (argv[++i] != NULL) {
                    strcpy(platformCfgName, argv[i]);
                }
            } else {
                LOG_ERR("%s: -p option is used but platform cfg name is not provided\n", __func__);
                return NVSIPL_STATUS_NOT_SUPPORTED;
            }
        } else if (!strcmp(argv[i], "-v")) {
            int loglevel = static_cast<int>(LEVEL_ERR);
            if (bDataAvailable) {
                loglevel = atoi(argv[++i]);
            } else {
                LOG_ERR("%s: -v option is provided without log level\n", __func__);
                return NVSIPL_STATUS_BAD_ARGUMENT;
            }
            switch (loglevel) {
            case LEVEL_INFO:
                LOG_LEVEL(LEVEL_INFO);
                break;
            case LEVEL_DBG:
                LOG_LEVEL(LEVEL_DBG);
                break;
            case LEVEL_WARN:
                LOG_LEVEL(LEVEL_WARN);
                break;
            default:
                LOG_LEVEL(LEVEL_ERR);
                break;
            }
#ifdef NVMEDIA_QNX
        } else if (!strcmp(argv[i], "-d")) {
            *bEnableDisp = true;
#endif // NVMEDIA_QNX
        } else {
            LOG_ERR("%s: %s is not a supported option\n", __func__, argv[i]);
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }

        i++;
    }

    if (i < argc) {
        LOG_ERR("%s: %s is not a supported option\n", __func__, argv[i]);
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    return NVSIPL_STATUS_OK;
}

#if NVMEDIA_QNX
static SIPLStatus GetDTPropAsString(const void* node,
                                    const char *const name,
                                    char val[],
                                    const uint32_t size)
{
    CHK_PTR_AND_RETURN_BADARG(node, "node");
    CHK_PTR_AND_RETURN_BADARG(name, "name");
    CHK_PTR_AND_RETURN_BADARG(val, "val");

    if (size == 0U) {
        LOG_ERR("size cannot be zero\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    uint32_t propLengthBytes = 0U;
    void const * const val_str = nvdt_node_get_prop(node, name, &propLengthBytes);
    CHK_PTR_AND_RETURN_BADARG(val_str, "val_str");
    if (propLengthBytes == 0U) {
        LOG_ERR("Property string cannot be zero-length\n");
        return NVSIPL_STATUS_ERROR;
    }

    if (propLengthBytes > size) {
        LOG_ERR("Property string exceeds maximum length\n");
        return NVSIPL_STATUS_ERROR;
    }

    memcpy(&val[0], val_str, static_cast<std::size_t>(propLengthBytes));

    if (val[(propLengthBytes-1U)] != '\0') {
        LOG_ERR("Failed to parse property string\n");
        return NVSIPL_STATUS_ERROR;
    }

    return NVSIPL_STATUS_OK;;
}
#endif // NVMEDIA_QNX

static SIPLStatus CheckSKU(const std::string &findStr, bool &bFound)
{
#if !NVMEDIA_QNX
    std::string sTargetModelNode = "/proc/device-tree/model";
    std::ifstream fs;
    fs.open(sTargetModelNode, std::ifstream::in);
    if (!fs.is_open()) {
        LOG_ERR("%s: cannot open board node %s\n", __func__, sTargetModelNode.c_str());
        return NVSIPL_STATUS_ERROR;
    }

    // Read the file in to the string.
    std::string nodeString;
    fs >> nodeString;

    if (strstr(nodeString.c_str(), findStr.c_str())) {
        bFound = true;
    }

    if (fs.is_open()) {
        fs.close();
    }
#else // NVMEDIA_QNX
    /* Get handle for DTB */
    if (NVDT_SUCCESS != nvdt_open()) {
        LOG_ERR("nvdt_open failed\n");
        return NVSIPL_STATUS_OUT_OF_MEMORY;
    }

    /* Check the Model */
    const void* modelNode;
    modelNode = nvdt_get_node_by_path("/");
    if (modelNode == NULL) {
        LOG_ERR("No node for model\n");
        (void) nvdt_close();
        return NVSIPL_STATUS_OUT_OF_MEMORY;
    }

    char name[20];
    SIPLStatus status = GetDTPropAsString(modelNode,
                                          "model",
                                          &name[0],
                                          static_cast<uint32_t>(sizeof(name) / sizeof(name[0])));
    if (status != NVSIPL_STATUS_OK) {
        (void) nvdt_close();
        return status;
    }

    if (strstr(name, findStr.c_str())) {
        bFound = true;
    }
    /* close nvdt once done */
    (void) nvdt_close();
#endif // !NVMEDIA_QNX

    return NVSIPL_STATUS_OK;
}

static SIPLStatus UpdatePlatformCfgPerBoardModel(PlatformCfg *platformCfg)
{
    CHK_PTR_AND_RETURN_BADARG(platformCfg, "platformCfg");

    /**
     * GPIO power control (GPIO7) is required for Drive Orin (P3663) but not
     * Firespray (P3710). GPIO0 is used for checking Error status.
     * If using another platform (something customer-specific, for example)
     * the GPIO field may need to be modified
     */
    bool isP3663 = false;
    std::vector<uint32_t> gpios;
    SIPLStatus status = CheckSKU("3663", isP3663);
    CHK_STATUS_AND_RETURN(status, "CheckSKU");
    if (isP3663) {
    /**
     * Bug 3951727: GPIO needs to be updated. Following values are taken from
     * Device tree files for QNX and Linux for P3663 and P3710.
     * checkSKU function checks for model in DT file and does a substring match.
     **/
#if !NVMEDIA_QNX
        gpios = {0, 1};       // For Linux in P3663
#else
        gpios = {0, 1, 7};    // For QNX in P3663
#endif
    } else {
        gpios = {0, 1};   // For Linux/QNX in P3710 and other boards
    }
    CHK_PTR_AND_RETURN_BADARG(platformCfg->deviceBlockList,
        "deviceBlockList");
    platformCfg->deviceBlockList[0].gpios = gpios;

    return status;
}

static SIPLStatus SiplMain(int argc, char *argv[])
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    PlatformCfg *platformCfg = nullptr;
    char platformCfgName[256] = "LI-OV2311-VCSEL-GMSL2-60H_DPHY_x4";
    const NvSIPLPipelineConfiguration *pipelineCfg = nullptr;
    NvSIPLPipelineConfiguration pipelineCfgIspOutputsEnabled = {
        .captureOutputRequested = true,
        .isp0OutputRequested = true,
        .isp1OutputRequested = true,
        .isp2OutputRequested = true,
        .disableSubframe = false
    };
    NvSIPLPipelineConfiguration pipelineCfgIspOutputsDisabled = {
        .captureOutputRequested = true,
        .isp0OutputRequested = false,
        .isp1OutputRequested = false,
        .isp2OutputRequested = false,
        .disableSubframe = false
    };
    NvSIPLPipelineQueues queues[MAX_NUM_SENSORS] {};
#ifdef NVMEDIA_QNX
    std::unique_ptr<CComposite> upComposite = nullptr;
#endif // NVMEDIA_QNX
    NvSciBufModule sciBufModule = nullptr;
    NvSciSyncModule sciSyncModule = nullptr;

    std::vector<uint8_t> blob;
    bool defaultNitoLoaded = true;
    std::mutex threadPrintMutex;
    ThreadData threadDataStructs[MAX_NUM_SENSORS][THREAD_INDEX_COUNT];
    bool quit = false;
    bool skipNito = false;

    std::unique_ptr<INvSIPLCamera> siplCamera = INvSIPLCamera::GetInstance();
    CHK_PTR_AND_RETURN(siplCamera, "INvSIPLCamera::GetInstance()");

    bool bEnableDisp = false;
    std::vector<uint32_t> vSensorIds;

    status = ParseArgs(argc, argv, platformCfgName, &bEnableDisp);
    if (status == NVSIPL_STATUS_EOF) {
        // PrintUsage() was called, application should exit
        return NVSIPL_STATUS_OK;
    } else {
        CHK_STATUS_AND_RETURN(status, "ParseArgs()");
    }

    CImageManager imageManager;
    LOG_INFO("platformCfg: %s\n", platformCfgName);
    if (!strcmp((platformCfgName), "LI-OV2311-VCSEL-GMSL2-60H_DPHY_x4")) {
        platformCfg = &platformCfgOv2311;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "V1SIM728S1RU3120NB20_CPHY_x4")) {
        platformCfg = &platformCfgIMX728VB2;
        pipelineCfg = &pipelineCfgIspOutputsEnabled;
        skipNito = false;
    } else if (!strcmp((platformCfgName), "V1SIM728S2RU3120HB30_CPHY_x4")) {
        platformCfg = &platformCfgIMX728VB3_1C;
        pipelineCfg = &pipelineCfgIspOutputsEnabled;
        skipNito = false;
    } else if (!strcmp((platformCfgName), "V1SIM728S2RU4120NB20_CPHY_x4")) {
        platformCfg = &platformCfgIMX728VB2_2;
        pipelineCfg = &pipelineCfgIspOutputsEnabled;
        skipNito = false;
    } else if (!strcmp((platformCfgName), "V1SIM623S3RU3200NB20_CPHY_x4")) {
        platformCfg = &platformCfgIMX623VB2;
        pipelineCfg = &pipelineCfgIspOutputsEnabled;
        skipNito = false;
    } else if (!strcmp((platformCfgName), "V1SIM623S4RU5195NB3_CPHY_x4")) {
        platformCfg = &platformCfgIMX623VB3_2;
        pipelineCfg = &pipelineCfgIspOutputsEnabled;
        skipNito = false;
    } else if (!strcmp((platformCfgName), "F008A120RM0A_CPHY_x4")) {
        platformCfg = &platformCfgAr0820;
        pipelineCfg = &pipelineCfgIspOutputsEnabled;
    } else if (!strcmp((platformCfgName), "IMX390_C_3461_F200_RGGB_CPHY_x4")) {
        platformCfg = &platformCfgImx390C;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "IMX390_C_3461_F200_RGGB_CPHY_x4_VCU")) {
        platformCfg = &platformCfgImx390C_VCU;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "AR0820C120FOV_24BIT_RGGB_CPHY_x2")) {
        platformCfg = &platformCfgAr0820B;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "VC0820C030R24_CPHY_x2")) {
        platformCfg = &platformCfgVC0820C030_A;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "VC0820C070R24_CPHY_x2")) {
        platformCfg = &platformCfgVC0820C070_A;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "VC0820C120R24_CPHY_x2")) {
        platformCfg = &platformCfgVC0820C120_A;
        pipelineCfg = &pipelineCfgIspOutputsEnabled;
        skipNito = false;
    } else if (!strcmp((platformCfgName), "OV2311_C_3461_CPHY_x2")) {
        platformCfg = &platformCfgOv2311C;
        // this module does not need to enable isp outputs.
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "OV2311_C_3461_CPHY_x2_VCU")) {
        platformCfg = &platformCfgOv2311C_VCU;
        // this module does not need to enable isp outputs.
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "VC0820C030R24_CPHY_x1_A_CUST1")) {
        platformCfg = &platformCfgVC0820C030_A_CUST1;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "VC0820C030R24_CPHY_x1_B_CUST1")) {
        platformCfg = &platformCfgVC0820C030_B_CUST1;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "VC0820C030R24_CPHY_x1_C_CUST1")) {
        platformCfg = &platformCfgVC0820C030_C_CUST1;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "VC0820C030R24_CPHY_x1_D_CUST1")) {
        platformCfg = &platformCfgVC0820C030_D_CUST1;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "VC0820C070R24_CPHY_x2_AB_CUST1")) {
        platformCfg = &platformCfgVC0820C070_AB_CUST1;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "VC0820C070R24_CPHY_x2_CD_CUST1")) {
        platformCfg = &platformCfgVC0820C070_CD_CUST1;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else if (!strcmp((platformCfgName), "VCC_GEN3_VC0820C070_CPHY_x2_VCU")) {
        platformCfg = &platformCfgVC0820C070_VCU;
        // currently for this module, isp outputs are set to be disabled and not load nito file
        pipelineCfg = &pipelineCfgIspOutputsDisabled;
        skipNito = true;
    } else {
        LOG_ERR("Unexpected platform configuration\n");
        return NVSIPL_STATUS_ERROR;
    }

    status = UpdatePlatformCfgPerBoardModel(platformCfg);
    CHK_STATUS_AND_RETURN(status, "UpdatePlatformCfgPerBoardModel");

    status = siplCamera->SetPlatformCfg(platformCfg);
    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::SetPlatformCfg()");

    // for each sensor
    for (auto d = 0U; d != platformCfg->numDeviceBlocks; d++) {
        auto db = platformCfg->deviceBlockList[d];
        for (auto m = 0U; m != db.numCameraModules; m++) {
            auto mod = db.cameraModuleInfoList[m];
            auto sensor = mod.sensorInfo;
            vSensorIds.push_back(sensor.id);
            status = siplCamera->SetPipelineCfg(sensor.id, *pipelineCfg, queues[sensor.id]);
            CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::SetPipelineCfg()");
        }
    }

    status = siplCamera->Init();
    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::Init()");

    auto sciStatus = NvSciBufModuleOpen(&sciBufModule);
    CHK_NVSCISTATUS_AND_RETURN(sciStatus, "NvSciBufModuleOpen");

    imageManager.Init(siplCamera.get(), *pipelineCfg, sciBufModule, bEnableDisp);
    CHK_STATUS_AND_RETURN(status, "CImageManager::Init()");
    for (const auto& uSensor : vSensorIds) {
        status = imageManager.Allocate(uSensor);
        CHK_STATUS_AND_RETURN(status, "CImageManager::Allocate()");
        status = imageManager.Register(uSensor);
        CHK_STATUS_AND_RETURN(status, "CImageManager::Register()");
    }

#ifdef NVMEDIA_QNX
    if (bEnableDisp) {
        LOG_INFO("Enable display.\n");

        upComposite.reset(new CComposite());
        CHK_PTR_AND_RETURN(upComposite, "Compositor creation");

        NvSciError sciErr = NvSciSyncModuleOpen(&sciSyncModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncModuleOpen");

        status = upComposite->Init(sciBufModule, sciSyncModule, &imageManager, vSensorIds);
        CHK_STATUS_AND_RETURN(status, "CComposite::Init()");
        NvMediaNvSciSyncClientType clientType2d[NUM_SYNC_INTERFACES] = { NVMEDIA_WAITER, NVMEDIA_SIGNALER };
        NvSiplNvSciSyncClientType clientTypeSipl[NUM_SYNC_INTERFACES] = { SIPL_SIGNALER, SIPL_WAITER };
        NvMediaNvSciSyncObjType syncObjType2d[NUM_SYNC_INTERFACES] = { NVMEDIA_PRESYNCOBJ, NVMEDIA_EOFSYNCOBJ };
        NvSiplNvSciSyncObjType syncObjTypeSipl[NUM_SYNC_INTERFACES] = { NVSIPL_EOFSYNCOBJ, NVSIPL_PRESYNCOBJ };
        for (uint32_t uSensor : vSensorIds) {
            // Only the ISP0 output is being displayed at this time
            if (pipelineCfg->isp0OutputRequested) {
                for (uint32_t i = 0U; i < NUM_SYNC_INTERFACES; i++) {
                    // Create attribute lists (with automatic destructors)
                    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrListSipl;
                    attrListSipl.reset(new NvSciSyncAttrList());
                    sciErr = NvSciSyncAttrListCreate(sciSyncModule, attrListSipl.get());
                    CHK_NVSCISTATUS_AND_RETURN(sciErr, "SIPL NvSciSyncAttrListCreate");

                    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrList2d;
                    attrList2d.reset(new NvSciSyncAttrList());
                    sciErr = NvSciSyncAttrListCreate(sciSyncModule, attrList2d.get());
                    CHK_NVSCISTATUS_AND_RETURN(sciErr, "2D NvSciSyncAttrListCreate");

                    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrListCpu;
                    attrListCpu.reset(new NvSciSyncAttrList());
                    sciErr = NvSciSyncAttrListCreate(sciSyncModule, attrListCpu.get());
                    CHK_NVSCISTATUS_AND_RETURN(sciErr, "CPU NvSciSyncAttrListCreate");

                    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> reconciledAttrList;
                    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> conflictAttrList;
                    reconciledAttrList.reset(new NvSciSyncAttrList());
                    conflictAttrList.reset(new NvSciSyncAttrList());

                    // Fill attribute lists
                    status = siplCamera->FillNvSciSyncAttrList(uSensor,
                                                               INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                                                               *attrListSipl,
                                                               clientTypeSipl[i]);
                    CHK_STATUS_AND_RETURN(status, "SIPL FillNvSciSyncAttrList");

                    status = upComposite->FillNvSciSyncAttrList(*attrList2d, clientType2d[i]);
                    CHK_STATUS_AND_RETURN(status, "2D FillNvSciSyncAttrList");

                    // Set CPU waiter attributes, they will get used later if necessary
                    NvSciSyncAttrKeyValuePair keyVals[2];
                    bool cpuAccess = true;
                    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
                    keyVals[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
                    keyVals[0].value = (void *)&cpuAccess;
                    keyVals[0].len = sizeof(cpuAccess);
                    keyVals[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
                    keyVals[1].value = (void*)&cpuPerm;
                    keyVals[1].len = sizeof(cpuPerm);

                    sciErr = NvSciSyncAttrListSetAttrs(*attrListCpu, keyVals, 2);
                    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");

                    // Reconcile attribute lists
                    NvSciSyncAttrList attrListsForReconcile[NUM_SYNC_ACTORS] = { *attrListSipl, *attrList2d, *attrListCpu };
                    sciErr = NvSciSyncAttrListReconcile(attrListsForReconcile,
                                                        NUM_SYNC_ACTORS,
                                                        reconciledAttrList.get(),
                                                        conflictAttrList.get());
                    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListReconcile");

                    // Allocate sync object
                    std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> syncObj;
                    syncObj.reset(new NvSciSyncObj());
                    sciErr = NvSciSyncObjAlloc(*reconciledAttrList, syncObj.get());
                    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjAlloc");

                    // Register sync object
                    status = siplCamera->RegisterNvSciSyncObj(uSensor,
                                                              INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                                                              syncObjTypeSipl[i],
                                                              *syncObj);
                    CHK_STATUS_AND_RETURN(status, "SIPL RegisterNvSciSyncObj");

                    status = upComposite->RegisterNvSciSyncObj(uSensor,
                                                               syncObjType2d[i],
                                                               std::move(syncObj));
                    CHK_STATUS_AND_RETURN(status, "2D RegisterNvSciSyncObj");
                }
            }
        }

        status = upComposite->Start();
        CHK_STATUS_AND_RETURN(status, "CComposite::Start()");
    }
#endif // NVMEDIA_QNX

    if (!skipNito) {
        if (!strcmp((platformCfgName), "AR0820C120FOV_24BIT_RGGB_CPHY_x2")) {
            status = LoadNitoFile(NITO_PATH, "AR0820C120FOV_24BIT_RGGB", blob, defaultNitoLoaded);
        } else if (!strcmp((platformCfgName), "VC0820C120R24_CPHY_x2")) {
            status = LoadNitoFile(NITO_PATH, "VC0820C120R24", blob, defaultNitoLoaded);
        } else if (!strcmp((platformCfgName), "VC0820C030R24_CPHY_x2")) {
            status = LoadNitoFile(NITO_PATH, "VC0820C030R24", blob, defaultNitoLoaded);
        } else if (!strcmp((platformCfgName), "VC0820C070R24_CPHY_x2")) {
            status = LoadNitoFile(NITO_PATH, "VC0820C070R24", blob, defaultNitoLoaded);
        } else if (!strcmp((platformCfgName), "IMX390_C_3461_F200_RGGB_CPHY_x4")) {
            status = LoadNitoFile(NITO_PATH, "IMX390_C_3461_F200_RGGB", blob, defaultNitoLoaded);
        } else if (!strcmp((platformCfgName), "V1SIM728S1RU3120NB20_CPHY_x4")) {
            status = LoadNitoFile(NITO_PATH, "V1SIM728S1RU3120NB20", blob, defaultNitoLoaded);
        } else if (!strcmp((platformCfgName), "V1SIM728S2RU3120HB30_CPHY_x4")) {
            status = LoadNitoFile(NITO_PATH, "V1SIM728S2RU3120HB30", blob, defaultNitoLoaded);
        } else if (!strcmp((platformCfgName), "V1SIM728S2RU4120NB20_CPHY_x4")) {
            status = LoadNitoFile(NITO_PATH, "V1SIM728S2RU4120NB20", blob, defaultNitoLoaded);
        } else if (!strcmp((platformCfgName), "V1SIM623S3RU3200NB20_CPHY_x4")) {
            status = LoadNitoFile(NITO_PATH, "V1SIM623S3RU3200NB20", blob, defaultNitoLoaded);
        } else if (!strcmp((platformCfgName), "V1SIM623S4RU5195NB3_CPHY_x4")) {
            status = LoadNitoFile(NITO_PATH, "V1SIM623S4RU5195NB3", blob, defaultNitoLoaded);
        } else if (!strcmp((platformCfgName), "F008A120RM0A_CPHY_x4")) {
            status = LoadNitoFile(NITO_PATH, "F008A120RM0A", blob, defaultNitoLoaded);
        } else {
            LOG_ERR("Unexpected platform configuration for NITO file\n");
            return NVSIPL_STATUS_ERROR;
        }

        CHK_STATUS_AND_RETURN(status, "LoadNitoFile()");
        // The NVIDIA plugin relies on the module-specific NITO and does not work with default.nito
        if (defaultNitoLoaded) {
            LOG_ERR("Module-specific NITO file not found\n");
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }

        for (const auto& uSensor : vSensorIds) {
            status = siplCamera->RegisterAutoControlPlugin(uSensor, NV_PLUGIN, nullptr, blob);
            CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterAutoControlPlugin()");
        }
    }
    /**
     * Check if pipelineCfg has required output requested as true to avoid
     * "Null callback or queue provided" error from QueueHandler start function
    **/
    for (const auto& uSensor : vSensorIds) {
        for (uint32_t i = 0U; i < THREAD_INDEX_COUNT; i++) {
            threadDataStructs[uSensor][i].printMutex = &threadPrintMutex;
            threadDataStructs[uSensor][i].uSensorId = uSensor;
        }
        if (pipelineCfg->captureOutputRequested) {
            threadDataStructs[uSensor][THREAD_INDEX_ICP].threadName =
                "ICP (Sensor:" + std::to_string(uSensor)+")";
            threadDataStructs[uSensor][THREAD_INDEX_ICP].imageQueueHandler.
                Start(
                    queues[uSensor].captureCompletionQueue,
                    &threadDataStructs[uSensor][THREAD_INDEX_ICP],
                    IMAGE_QUEUE_TIMEOUT_US
                );
        }
        if (pipelineCfg->isp0OutputRequested) {
#ifdef NVMEDIA_QNX
            if (upComposite != nullptr) {
                threadDataStructs[uSensor][THREAD_INDEX_ISP0].pComposite =
                    upComposite.get();
            }
#endif // NVMEDIA_QNX
            threadDataStructs[uSensor][THREAD_INDEX_ISP0].threadName =
                "ISP0(Sensor:" + std::to_string(uSensor)+")";
            threadDataStructs[uSensor][THREAD_INDEX_ISP0].imageQueueHandler.
                Start(
                    queues[uSensor].isp0CompletionQueue,
                    &threadDataStructs[uSensor][THREAD_INDEX_ISP0],
                    IMAGE_QUEUE_TIMEOUT_US
                );
            }
        if (pipelineCfg->isp1OutputRequested) {
            threadDataStructs[uSensor][THREAD_INDEX_ISP1].threadName =
                "ISP1(Sensor:" + std::to_string(uSensor)+")";
            threadDataStructs[uSensor][THREAD_INDEX_ISP1].imageQueueHandler.
                Start(
                    queues[uSensor].isp1CompletionQueue,
                    &threadDataStructs[uSensor][THREAD_INDEX_ISP1],
                    IMAGE_QUEUE_TIMEOUT_US
                );
        }
        if (pipelineCfg->isp2OutputRequested) {
            threadDataStructs[uSensor][THREAD_INDEX_ISP2].threadName =
                "ISP2(Sensor:" + std::to_string(uSensor)+")";
            threadDataStructs[uSensor][THREAD_INDEX_ISP2].imageQueueHandler.
                Start(
                    queues[uSensor].isp2CompletionQueue,
                    &threadDataStructs[uSensor][THREAD_INDEX_ISP2],
                    IMAGE_QUEUE_TIMEOUT_US
                );
        }
        threadDataStructs[uSensor][THREAD_INDEX_EVENT].threadName =
            "Event(Sensor:" + std::to_string(uSensor)+")";
        threadDataStructs[uSensor][THREAD_INDEX_EVENT].eventQueueHandler.
            Start(
                queues[uSensor].notificationQueue,
                &threadDataStructs[uSensor][THREAD_INDEX_EVENT],
                EVENT_QUEUE_TIMEOUT_US
            );
    }

    status = siplCamera->Start();
    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::Start()");

    // Wait for the user's quit command
    while (!quit) {
        std::cout << "Enter 'q' to quit the application\n-\n";
        char line[INPUT_LINE_READ_SIZE];
        std::cin.getline(line, INPUT_LINE_READ_SIZE);
        if (line[0] == 'q') {
            quit = true;
        }
    }

    status = siplCamera->Stop();
    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::Stop()");

#ifdef NVMEDIA_QNX
    if (upComposite != nullptr) {
        status = upComposite->Stop();
        CHK_STATUS_AND_RETURN(status, "CComposite::Stop()");
    }
#endif // NVMEDIA_QNX

    for (const auto& uSensor : vSensorIds) {
        for (uint32_t i = 0U; i < THREAD_INDEX_COUNT; i++) {
            if (threadDataStructs[uSensor][i].imageQueueHandler.IsRunning()) {
                threadDataStructs[uSensor][i].imageQueueHandler.Stop();
            }
            if (threadDataStructs[uSensor][i].eventQueueHandler.IsRunning()) {
                threadDataStructs[uSensor][i].eventQueueHandler.Stop();
            }
        }
        std::cout << "Sensor" << uSensor << "\t\t" \
                  << "Frame drops: " \
                  << threadDataStructs[uSensor][THREAD_INDEX_EVENT].uNumFrameDrops << "\t\t" \
                  << "Frame discontinuities: " \
                  << threadDataStructs[uSensor][THREAD_INDEX_EVENT].uNumFrameDiscontinuities \
                  << std::endl;
    }

#ifdef NVMEDIA_QNX
    if (upComposite != nullptr) {
        status = upComposite->Deinit();
        CHK_STATUS_AND_RETURN(status, "CComposite::Deinit()");
    }
#endif // NVMEDIA_QNX

    status = siplCamera->Deinit();
    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::Deinit()");

    if (sciSyncModule != nullptr) {
        NvSciSyncModuleClose(sciSyncModule);
    }

    if (sciBufModule != nullptr) {
        NvSciBufModuleClose(sciBufModule);
    }

    std::cout << "SUCCESS!" << std::endl;
    return status;
}

int main(int argc, char *argv[])
{
    SIPLStatus status = SiplMain(argc, argv);
    return static_cast<int>(status);
}
