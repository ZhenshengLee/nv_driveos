/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <iostream>
#include <chrono>
#include <pthread.h>

#include "CComposite.hpp"
#include "CUtils.hpp"
#include "CNvWfd.hpp"

using namespace std;

SIPLStatus CComposite::CreateHelpers(uint32_t uNumDisplays) {
    for (uint32_t i = 0U; i < uNumDisplays; i++) {
        m_helpers[i].reset(new CCompositeHelper<BufferItem>(m_groupInfos));
        if (m_helpers[i] ==  nullptr) {
            LOG_ERR("Failed to create CCompositeHelper\n");
            return NVSIPL_STATUS_OUT_OF_MEMORY;
        }
    }
    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::InitDisplay(uint32_t uNumDisplays) {
    m_dispMgr.reset(new CNvWfdResourcesCommon());
    if (m_dispMgr == nullptr) {
        LOG_ERR("IDisplayManager memory allocation failed\n");
        return NVSIPL_STATUS_OUT_OF_MEMORY;
    }
    SIPLStatus status = m_dispMgr->Init(uNumDisplays);
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("Unable to initialize IDisplayManager\n");
        return status;
    }
    for (uint32_t i = 0U; i < uNumDisplays; i++) {
        if (m_helpers[i] == nullptr) {
            LOG_ERR("Issue with compositor helper\n");
            return NVSIPL_STATUS_ERROR;
        }
        IDisplayInterface *pDispIf = nullptr;
        status = m_dispMgr->GetDisplayInterface(i, pDispIf);
        if ((status != NVSIPL_STATUS_OK) || (pDispIf == nullptr)) {
            LOG_ERR("Failed to get display interface\n");
            // Overwrite status if error was due to null pointer
            status = (status == NVSIPL_STATUS_OK) ? NVSIPL_STATUS_ERROR : status;
            return status;
        }
        m_helpers[i]->m_dispIf = pDispIf;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::Init(uint32_t uNumDisplays,
                            NvSiplRect *pRect,
                            NvSciBufModule bufModule,
                            NvSciSyncModule syncModule)
{
    if (uNumDisplays > MAX_SUPPORTED_DISPLAYS) {
        LOG_ERR("Compositor: Invalid number of displays requested\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if ((bufModule == nullptr) || (syncModule == nullptr)) {
        LOG_ERR("Compositor: Received unexpected nullptr\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    // Initialize member variables
    m_sciBufModule = bufModule;
    m_sciSyncModule = syncModule;
    m_bRunning = false;

    NvSciError sciErr = NvSciSyncCpuWaitContextAlloc(syncModule, &m_cpuWaitContext);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");

    SIPLStatus status = CreateHelpers(uNumDisplays);
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("Compositor: Failed to create helpers\n");
        return status;
    }

    // Initialize display
    status = InitDisplay(uNumDisplays);
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("Compositor: InitDisplay failed\n");
        return status;
    }

    for (uint32_t g = 0U; g < NUM_OF_GROUPS; g++) {
        m_iGroupInfos[g]->m_bGroupInUse = false;
        m_iGroupInfos[g]->m_id = g;
    }

    uint32_t g = 0U;
    for (uint32_t i = 0U; i < uNumDisplays; i++) {
        status = m_helpers[i]->Init(pRect, m_sciBufModule, m_sciSyncModule);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Compositor: Init failed\n");
            return status;
        }
        while ((g < NUM_OF_GROUPS) && (!m_iGroupInfos[g]->HasInputs())) {
            g++;
        }
        if (g == NUM_OF_GROUPS) {
            LOG_ERR("Compositor: Could not find available group for helper\n");
            return NVSIPL_STATUS_ERROR;
        }
        // Indicate that group is in use
        bool bGroupInUse = false;
        if (!m_iGroupInfos[g]->m_bGroupInUse.compare_exchange_strong(bGroupInUse, true)) {
            LOG_ERR("Compositor: Group is already in use:%u\n", g);
            return NVSIPL_STATUS_ERROR;
        }
        // Assign group ID for helper
        m_helpers[i]->m_uGroupIndex = g;
        m_helpers[i]->m_uNewGroupIndex = g;
        g++;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::RegisterSource(uint32_t groupIndex,
                                      uint32_t modIndex,
                                      uint32_t outIndex,
                                      bool isRaw,
                                      uint32_t &id,
                                      bool isSimulatorMode,
                                      NvSciStreamBlock *consumer,
                                      NvSciStreamBlock **upstream,
                                      NvSciStreamBlock *queue,
                                      QueueType type)
{
    if ((groupIndex >= NUM_OF_GROUPS) or (modIndex >= NUM_OF_ROWS) or (outIndex >= NUM_OF_COLS)) {
        LOG_ERR("Compositor: RegisterSource: Invalid argument\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    // Compute global ID for the source
    id = groupIndex * (NUM_OF_COLS * NUM_OF_ROWS) + modIndex * NUM_OF_COLS + outIndex;

    SIPLStatus status = m_iGroupInfos[groupIndex]->AddInput(id, isSimulatorMode);
    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    LOG_INFO("Compositor: Registered output:%u from link:%u of quad:%u as id:%u\n", outIndex, modIndex, groupIndex, id);
    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::FillNvSciSyncAttrList(uint32_t id,
                                             NvSciSyncAttrList &attrList,
                                             NvMediaNvSciSyncClientType clientType)
{
    if (m_helpers[0] == nullptr) {
        LOG_ERR("Compositor: No helper available to fill NvSciSyncAttrList\n");
        return NVSIPL_STATUS_INVALID_STATE;
    }
    return m_helpers[0]->FillNvSciSyncAttrList(id, attrList, clientType);
}

SIPLStatus CComposite::RegisterNvSciSyncObj(uint32_t id,
                                            NvMediaNvSciSyncObjType syncObjType,
                                            std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> syncObj)
{
    // Get the input matching the requested ID
    CInputSync *pInput = nullptr;
    for (uint32_t g = 0U; g < NUM_OF_GROUPS; g++) {
        if (m_iGroupInfos[g]->GetInput(id, pInput)) {
            break;
        }
    }
    if (pInput == nullptr) {
        LOG_ERR("Could not find matching input for ID: %u\n", id);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    // Register this sync object with all helpers since any stream can get sent to any display
    for (uint32_t i = 0U; i < MAX_SUPPORTED_DISPLAYS; i++) {
        if (m_helpers[i]  != nullptr) {
            SIPLStatus status = m_helpers[i]->RegisterNvSciSyncObj(id, syncObjType, *syncObj);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Compositor: RegisterNvSciSyncObj failed for source:%u\n", id);
                return status;
            }
        }
    }

    // Assign ownership of the sync object to the input
    pInput->SetSyncObj(std::move(syncObj), syncObjType);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::UnregisterNvSciSyncObjs()
{
    SIPLStatus ret = NVSIPL_STATUS_OK;
    std::vector<CInputSync *> pInputs {nullptr};
    for (uint32_t g = 0U; g < NUM_OF_GROUPS; g++) {
        m_iGroupInfos[g]->GetAllInputs(pInputs);
        for (CInputSync * &pInput : pInputs) {
            // Unregister this sync object with all helpers
            for (uint32_t i = 0U; i < MAX_SUPPORTED_DISPLAYS; i++) {
                if (m_helpers[i]  != nullptr) {
                    NvSciSyncObj syncObj;
                    NvMediaNvSciSyncObjType syncObjTypes[2] = { NVMEDIA_PRESYNCOBJ, NVMEDIA_EOFSYNCOBJ };
                    for (uint32_t j = 0U; j < (sizeof(syncObjTypes)/sizeof(syncObjTypes[0])); j++) {
                        SIPLStatus status = pInput->GetSyncObj(syncObj, syncObjTypes[j]);
                        CHK_STATUS_AND_RETURN(status, "GetSyncObj");
                        status = m_helpers[i]->UnregisterNvSciSyncObj(pInput->m_id, syncObj);
                        if (status != NVSIPL_STATUS_OK) {
                            LOG_ERR("Compositor: UnregisterNvSciSyncObj failed for source:%u\n", pInput->m_id);
                            ret = status;
                        }
                    }
                }
            }
        }
    }

    return ret;
}

SIPLStatus CComposite::FillNvSciBufAttrList(NvSciBufAttrList &attrList)
{
    if (m_helpers[0] == nullptr) {
        LOG_ERR("Compositor: No helper available to fill NvSciBufAttrList\n");
        return NVSIPL_STATUS_INVALID_STATE;
    }
    return m_helpers[0]->FillNvSciBufAttrList(attrList);
}

SIPLStatus CComposite::RegisterNvSciBufObj(uint32_t id, NvSciBufObj &bufObj)
{
    // Get the input matching the requested ID
    CInputBuffers *pInput = nullptr;
    for (uint32_t g = 0U; g < NUM_OF_GROUPS; g++) {
        if (m_iGroupInfos[g]->GetInput(id, pInput)) {
            break;
        }
    }
    if (pInput == nullptr) {
        LOG_ERR("Could not find matching input for ID: %u\n", id);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    NvSciBufObj bufObjToRegister = bufObj;
    // Check if buffer is raw and allocate scratch RGBA surface if necessary
    bool isRawBuffer = false;
    SIPLStatus status = CUtils::IsRawBuffer(bufObj, isRawBuffer);
    CHK_STATUS_AND_RETURN(status, "CUtils::IsRawBuffer");
    if (isRawBuffer) {
        if (pInput->m_rgbaRegisteredWith2d) {
            // RGBA buffer was previously allocated and registered, no further action is required
            return NVSIPL_STATUS_OK;
        }
        NvSciBufAttrList bufAttrList;
        NvSciError sciErr = NvSciBufAttrListCreate(m_sciBufModule, &bufAttrList);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate");
        NvSciBufType bufType = NvSciBufType_Image;
        NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
        NvSciBufAttrKeyValuePair attrKvp[] = {
            { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) }
        };
        sciErr = NvSciBufAttrListSetAttrs(bufAttrList, attrKvp, 2U);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
        SIPLStatus status = FillNvSciBufAttrList(bufAttrList);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Compositor: FillNvSciBufAttrList failed for source:%u\n", id);
            return status;
        }
        status = pInput->AllocateImageBuffers(m_sciBufModule, bufObj, bufAttrList);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Compositor: AllocateImageBuffers failed for source:%u\n", id);
            return status;
        }
        status = pInput->GetRGBABuffer(bufObjToRegister);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Compositor: GetRGBABuffer failed for source:%u\n", id);
            return status;
        }
    }

    // Register the buffer with all helpers since any stream can get sent to any display
    for (uint32_t i = 0U; i < MAX_SUPPORTED_DISPLAYS; i++) {
        if (m_helpers[i]  != nullptr) {
            SIPLStatus status = m_helpers[i]->RegisterNvSciBufObj(id, bufObjToRegister);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Compositor: RegisterNvSciBufObj failed for source:%u\n", id);
                return status;
            }
            if (isRawBuffer) {
                pInput->m_rgbaRegisteredWith2d = true;
            }
        }
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::UnregisterNvSciBufObj(uint32_t id, NvSciBufObj &bufObj)
{
    // Get the input matching the requested ID
    CInputBuffers *pInput = nullptr;
    for (uint32_t g = 0U; g < NUM_OF_GROUPS; g++) {
        if (m_iGroupInfos[g]->GetInput(id, pInput)) {
            break;
        }
    }
    if (pInput == nullptr) {
        LOG_ERR("Could not find matching input for ID\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    NvSciBufObj bufObjToUnregister = bufObj;
    bool isRawBuffer = false;
    SIPLStatus status = CUtils::IsRawBuffer(bufObj, isRawBuffer);
    CHK_STATUS_AND_RETURN(status, "CUtils::IsRawBuffer");
    if (isRawBuffer) {
        if (!pInput->m_rgbaRegisteredWith2d) {
            // RGBA buffer was previously unregistered, no further action is required
            return NVSIPL_STATUS_OK;
        }
        SIPLStatus status = pInput->GetRGBABuffer(bufObjToUnregister);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Compositor: GetRGBABuffer failed for source:%u\n", id);
            return status;
        }
    }

    // Unregister the buffer with all helpers
    for (uint32_t i = 0U; i < MAX_SUPPORTED_DISPLAYS; i++) {
        if (m_helpers[i]  != nullptr) {
            SIPLStatus status = m_helpers[i]->UnregisterNvSciBufObj(id, bufObjToUnregister);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Compositor: UnregisterNvSciBufObj failed for source:%u\n", id);
                return status;
            }
            if (isRawBuffer) {
                pInput->m_rgbaRegisteredWith2d = false;
            }
        }
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::Post(uint32_t id, INvSIPLClient::INvSIPLNvMBuffer *pBuffer)
{
    if (!m_bRunning) {
        // Composite is not ready to accept buffers
        LOG_WARN("Compositor is not ready to accept buffers\n");
        return NVSIPL_STATUS_OK;
    }

    // Check if ID belongs to a group that is currently active
    CInputInfo<BufferItem> *pInput = nullptr;
    for (uint32_t g = 0U; g < NUM_OF_GROUPS; g++) {
        if (m_groupInfos[g].GetIfActive(id, pInput)) {
            break;
        }
    }
    if (pInput == nullptr) {
        // Source does not belong to an active group
        return NVSIPL_STATUS_OK;
    }

    // Add buffer to queue
    BufferItem item = BufferItem(pBuffer, (pInput->m_isSimulatorMode ? &m_cpuWaitContext : nullptr));
    pInput->QueueAdd(item);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::SetActiveGroup(uint32_t dispId, uint32_t groupIndex)
{
    if ((dispId >= MAX_SUPPORTED_DISPLAYS) || (m_helpers[dispId] == nullptr)) {
        LOG_ERR("Compositor: Invalid display ID:%u\n", dispId);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (groupIndex >= NUM_OF_GROUPS) {
        LOG_ERR("Compositor: Invalid quad number:%u\n", groupIndex);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (!m_iGroupInfos[groupIndex]->HasInputs()) {
        LOG_ERR("Compositor: No displayable outputs in group:%u\n", groupIndex);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    // Mark group as being in use
    bool bGroupInUse = false;
    if (!m_iGroupInfos[groupIndex]->m_bGroupInUse.compare_exchange_strong(bGroupInUse, true)) {
        LOG_ERR("Compositor: Group is already in use:%u\n", groupIndex);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    uint32_t currentGroupIndex = m_helpers[dispId]->m_uGroupIndex;
    // Update group index
    if (!m_helpers[dispId]->m_uNewGroupIndex.compare_exchange_strong(currentGroupIndex, groupIndex)) {
        LOG_ERR("Compositor: Previous group change has not taken effect yet:%u\n", groupIndex);
        // Change did not take effect, release ownership of group
        bGroupInUse = true;
        if (!m_iGroupInfos[groupIndex]->m_bGroupInUse.compare_exchange_strong(bGroupInUse, false)) {
            LOG_ERR("Compositor: Failed to release group ownership:%u\n", groupIndex);
            return NVSIPL_STATUS_ERROR;
        }
        return NVSIPL_STATUS_ERROR;
    }

    return NVSIPL_STATUS_OK;
}

void CComposite::PrintDisplayableGroups() const
{
    string s[NUM_OF_GROUPS] = { "PORT-A", "PORT-B", "PORT-C", "PORT-D" };
    for (uint32_t g = 0U; g < NUM_OF_GROUPS; g++) {
        if (m_iGroupInfos[g]->HasInputs()) {
            cout << g << ":\t" << s[g] << endl;
        }
    }
}

SIPLStatus CComposite::Start()
{
    for (uint32_t i = 0U; i < MAX_SUPPORTED_DISPLAYS; i++) {
        if (m_helpers[i] != nullptr) {
            // Call start for helper
            SIPLStatus status = m_helpers[i]->Start();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Compositor: Start failed\n");
                return status;
            }
        }
    }
    m_bRunning = true;

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::Stop()
{
    m_bRunning = false;
    // Call stop for all helpers
    for (uint32_t i = 0U; i < MAX_SUPPORTED_DISPLAYS; i++) {
        if (m_helpers[i] != nullptr) {
            SIPLStatus status = m_helpers[i]->Stop();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Compositor: Stop failed\n");
                return status;
            }
        }
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::Deinit()
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    SIPLStatus ret = NVSIPL_STATUS_OK;

    // Call deinitialization for all helpers
    for (uint32_t i = 0U; i < MAX_SUPPORTED_DISPLAYS; i++) {
        if (m_helpers[i] != nullptr) {
            status = m_helpers[i]->Deinit();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Compositor: Deinit failed\n");
                ret = status;
            }
        }
    }

    // Dequeue and release all input buffers
    for (uint32_t g = 0U; g < NUM_OF_GROUPS; g++) {
        m_iGroupInfos[g]->DequeueAndReleaseAll();
    }

    // Unregister all NvSciSyncObjs
    status = UnregisterNvSciSyncObjs();
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("Compositor: UnregisterNvSciSyncObjs failed\n");
        ret = status;
    }

    NvSciSyncCpuWaitContextFree(m_cpuWaitContext);

    m_bDeinitialized = true;

    return ret;
}
