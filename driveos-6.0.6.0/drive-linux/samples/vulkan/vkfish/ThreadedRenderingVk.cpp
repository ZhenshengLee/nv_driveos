//----------------------------------------------------------------------------------
// File:        vk10-kepler\ThreadedRenderingVk/ThreadedRenderingVk.cpp
// SDK Version: v3.00 
// Email:       gameworks@nvidia.com
// Site:        http://developer.nvidia.com/
//
// Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//----------------------------------------------------------------------------------
#include "ThreadedRenderingVk.h"
#include "NvInstancedModelExtVK.h"

#include "NvAssetLoader.h"
#include "NvModelExt.h"
#include "NvVkContext.h"
#include "NvModelExtVK.h"
#include "NvQuadVK.h"
#include "NvLogs.h"
#include "NvAssert.h"

#include <stdint.h>

#include <ctime>

#define ARRAY_SIZE(a) ( sizeof(a) / sizeof( (a)[0] ))

#define NV_UNUSED( variable ) ( void )( variable )

// Currently the number of instances rendered of each model
#define MAX_INSTANCE_COUNT 100
#define INSTANCE_COUNT 100

#ifdef ANDROID
#define MAX_SCHOOL_COUNT 2000
#else
#define MAX_SCHOOL_COUNT 5000
#endif

#define SCHOOL_COUNT 50 // one of each fish model

extern uint32_t neighborOffset;
extern uint32_t neighborSkip;

#define EMBEDDED_RESOURCES 1


#ifdef _WIN32
DWORD WINAPI ThreadJobFunctionThunk(VOID *arg)
#else
void* ThreadJobFunctionThunk(void *arg);

void* ThreadJobFunctionThunk(void *arg)
#endif
{
    ThreadedRenderingVk::ThreadData* data = (ThreadedRenderingVk::ThreadData*)arg;
    data->m_app->ThreadJobFunction(data->m_index);

    return 0;
}

void ThreadedRenderingVk::ThreadJobFunction(uint32_t threadIndex)
{
    //NvThreadManager* threadManager = getThreadManagerInstance();
    //NV_ASSERT(NULL != threadManager);
    ThreadData& thread = m_Threads[threadIndex];
    thread.m_frameID = 0;

    while (m_running) {
        m_FrameStartLock->lockMutex();
        {
            if (m_frameID == thread.m_frameID)    {
                m_FrameStartCV->waitConditionVariable(
                    m_FrameStartLock);
            }

            thread.m_frameID = m_frameID;

            if (!m_running) {
                m_FrameStartLock->unlockMutex();
                break;
            }
            if (threadIndex >= m_activeThreads)    {
                m_FrameStartLock->unlockMutex();
                continue;
            }

        }

        m_FrameStartLock->unlockMutex();
        uint32_t schoolsDone = 0;

        {
            VkResult result;
            //CPU_TIMER_SCOPE(CPU_TIMER_THREAD_BASE_TOTAL + threadIndex);

            ThreadData& me = m_Threads[threadIndex];

            schoolsDone = me.m_schoolCount;
            uint32_t schoolMax = me.m_baseSchoolIndex + schoolsDone;


                VkCommandBuffer cmd = m_subCommandBuffers[threadIndex + 1];

                for (uint32_t i = me.m_baseSchoolIndex; i < schoolMax; i++) {
                    
                        if (!m_animPaused)
                            UpdateSchool(threadIndex, i, m_schools[i]);
                    

                    
                        if (m_threadedRendering) {
                            thread.m_drawCallCount += DrawSchool(cmd, m_schools[i], !thread.m_cmdBufferOpen);
                            thread.m_cmdBufferOpen = true;
                        }
                    
                }

                if (thread.m_cmdBufferOpen) {
                    //printf("thread render:%d\n", threadIndex);
                    result = vkEndCommandBuffer(cmd);
                    CHECK_VK_RESULT();
                }
            

            {
                m_DoneCountLock->lockMutex();
                {
                    m_doneCount += schoolsDone;
                    m_DoneCountCV->signalConditionVariable();
                }
                m_DoneCountLock->unlockMutex();
            }
        }
    }

    LOGI("Thread %d Exit.\n", threadIndex);
}

class ThreadedRenderingModelLoader : public Nv::NvModelFileLoader
{
public:
    ThreadedRenderingModelLoader() {}
    virtual ~ThreadedRenderingModelLoader() {}
    virtual char* LoadDataFromFile(const char* fileName)
    {
        int32_t length;
        return NvAssetLoaderRead(fileName, length);
    }

    virtual void ReleaseData(char* pData)
    {
        NvAssetLoaderFree(pData);
    }
};
//-----------------------------------------------------------------------------
// PUBLIC METHODS, CTOR AND DTOR
nv::vec3f ThreadedRenderingVk::ms_tankMin(-30.0f, 5.0f, -30.0f);
nv::vec3f ThreadedRenderingVk::ms_tankMax(30.0f, 25.0f, 30.0f);

ThreadedRenderingVk::ThreadedRenderingVk() :
    m_requestedActiveThreads(MAX_THREAD_COUNT),
    m_activeThreads(0),
    m_requestedThreadedRendering(true),
    m_threadedRendering(true),
    m_running(true),
    m_queueMutexPtr(NULL),
    m_startingCameraPitchYaw(0.0f, 0.0f),
    m_maxSchools(MAX_SCHOOL_COUNT),
    m_schoolStateMgr(m_maxSchools),
    m_activeSchools(0),
    m_causticTiling(0.1f),
    m_causticSpeed(0.3f),
    m_uiResetMode(RESET_RANDOM),
    m_animPaused(false),
    m_avoidance(true),
    m_currentTime(0.0f),
    m_frameLogicalClock(0),
    m_uiTankSize(30),
    m_uiInstanceCount(INSTANCE_COUNT),
    m_uiBatchSizeRequested(m_uiInstanceCount),
    m_uiBatchSizeActual(m_uiInstanceCount),
    m_uiSchoolInfoId(0),
    m_uiCameraFollow(false),
    m_uiSchoolDisplayModelIndex(0),
    m_drawCallCount(0),
    m_statsCountdown(STATS_FRAMES),
    m_statsMode(STATS_SIMPLE),
    m_flushPerFrame(false),
    m_frameID(0)
{
    ms_tankMax.x = ms_tankMax.z = (float)m_uiTankSize;
    ms_tankMin.x = ms_tankMin.z = -ms_tankMax.x;
    m_startingCameraPosition = (ms_tankMin + ms_tankMax) * 0.5f;
    m_startingCameraPosition.z += 40.0f;

    for (uint32_t i = 0; i < MODEL_COUNT; i++)
    {
        m_models[i] = NULL;
    }

    m_FrameStartLock = NULL;
    m_FrameStartCV = NULL;
    m_NeedsUpdateQueueLock = NULL;
    m_DoneCountLock = NULL;
    m_DoneCountCV = NULL;

    //ool m_running = true;

    for (uint32_t i = 0; i < MAX_THREAD_COUNT; i++)
    {
        m_Threads[i].m_thread = NULL;
        m_Threads[i].m_app = this;
        m_Threads[i].m_index = i;
    }

    InitializeSchoolDescriptions(50);

    // Required in all subclasses to avoid silent link issues
    forceLinkHack();

    m_initSchools = SCHOOL_COUNT;

    {
        const std::vector<std::string>& cmd = getCommandLine();
        for (std::vector<std::string>::const_iterator iter = cmd.begin(); iter != cmd.end(); ++iter)
        {
            if (*iter == "-idle")
                m_flushPerFrame = true;
            else if (*iter == "-schools") {
                iter++;
                if (iter == cmd.end()) {
                    break;
                }
                else {
                    m_initSchools = atoi(iter->c_str());
                }
            }
        }
    }

    m_initSchools = (m_initSchools > MAX_SCHOOL_COUNT) ? MAX_SCHOOL_COUNT : m_initSchools;
}

ThreadedRenderingVk::~ThreadedRenderingVk()
{
    LOGI("ThreadedRenderingVk: destroyed\n");
}

// Inherited methods
ThreadedRenderingModelLoader loader;

#if defined(EMBEDDED_RESOURCES)
    #include "assets/textures/sand_dds.h"
    #include "assets/textures/caustic1_dds.h"
    #include "assets/textures/caustic2_dds.h"
    #include "assets/textures/gradient_dds.h"
#endif

void ThreadedRenderingVk::initRendering(void)
{
    NV_APP_BASE_SHARED_INIT();
    m_width = 1920;
    m_height = 1080;

    /*for (int32_t i = 0; i < CPU_TIMER_COUNT; ++i)
    {
        m_CPUTimers[i].init();
    }*/

    //mFramerate->setReportFrames(20);

    VkResult result;
    NvAssetLoaderAddSearchPath(".");
    Nv::NvModelExt::SetFileLoader(&loader);

    NV_ASSERT(NULL == m_queueMutexPtr);
    m_queueMutexPtr = getThreadManagerInstance()->initializeMutex(true,
        NvMutex::MutexLockLevelInitial);
    NV_ASSERT(NULL != m_queueMutexPtr);

    // Load all shaders
    VkPipelineShaderStageCreateInfo fishShaderStages[2];
    uint32_t shaderCount = 0;
#ifdef SOURCE_SHADERS
    shaderCount = vk().createShadersFromSourceString(
        NvAssetLoadTextFile("src_shaders/staticfish.glsl"), fishShaderStages, 2);
#else
    {
        int32_t length;

#if defined(EMBEDDED_RESOURCES)
        #include "assets/shaders/staticfish_nvs.h"
        char* data = (char*) malloc(staticfish_nvs_len);
        memcpy(data, staticfish_nvs, staticfish_nvs_len);
        length = staticfish_nvs_len;

#else
        char* data = NvAssetLoaderRead("shaders/staticfish.nvs", length);
#endif
        shaderCount = vk().createShadersFromBinaryBlob((uint32_t*)data,
            length, fishShaderStages, 2);
    }
#endif

    if (shaderCount == 0) {
        showDialog("Fatal: Cannot Find Assets", "The shader assets cannot be loaded.\n"
            "Please ensure that the assets directory for the sample is in place\n"
            "and has not been moved.  Exiting.", true);
        return;
    }

    VkPipelineShaderStageCreateInfo skyShaderStages[2];
#ifdef SOURCE_SHADERS
    shaderCount = vk().createShadersFromSourceString(
    NvAssetLoadTextFile("src_shaders/skyboxcolor.glsl"), skyShaderStages, 2);
#else
    {
        int32_t length;

#if defined(EMBEDDED_RESOURCES)
        #include "assets/shaders/skyboxcolor_nvs.h"
        char* data = (char*) malloc(skyboxcolor_nvs_len);
        memcpy(data, skyboxcolor_nvs, skyboxcolor_nvs_len);
        length = skyboxcolor_nvs_len;

#else
        char* data = NvAssetLoaderRead("shaders/skyboxcolor.nvs", length);
#endif
        shaderCount = vk().createShadersFromBinaryBlob((uint32_t*)data,
            length, skyShaderStages, 2);
    }
#endif

    if (shaderCount == 0) {
        showDialog("Fatal: Cannot Find Assets", "The shader assets cannot be loaded.\n"
            "Please ensure that the assets directory for the sample is in place\n"
            "and has not been moved.  Exiting.", true);
        return;
    }

    VkPipelineShaderStageCreateInfo groundShaderStages[2];
#ifdef SOURCE_SHADERS
    shaderCount = vk().createShadersFromSourceString(
        NvAssetLoadTextFile("src_shaders/groundplane.glsl"), groundShaderStages, 2);
#else
    {
        int32_t length;
#if defined(EMBEDDED_RESOURCES)
        #include "assets/shaders/groundplane_nvs.h"
        char* data = (char*) malloc(groundplane_nvs_len);
        memcpy(data, groundplane_nvs, groundplane_nvs_len);
        length = groundplane_nvs_len;
#else
        char* data = NvAssetLoaderRead("shaders/groundplane.nvs", length);
#endif
        shaderCount = vk().createShadersFromBinaryBlob((uint32_t*)data,
            length, groundShaderStages, 2);
    }
#endif

    if (shaderCount == 0) {
        showDialog("Fatal: Cannot Find Assets", "The shader assets cannot be loaded.\n"
            "Please ensure that the assets directory for the sample is in place\n"
            "and has not been moved.  Exiting.", true);
        return;
    }

    // Create the standard samplers that will be used
    InitializeSamplers();

    for (uint32_t i = 0; i < MODEL_COUNT; ++i)
    {
#if defined(EMBEDDED_RESOURCES)
        Nv::NvModelExt* pModel =
            Nv::NvModelExt::CreateFromData(ms_modelInfo[i].m_dataBuf);
#else
        Nv::NvModelExt* pModel =
            Nv::NvModelExt::CreateFromPreprocessed(ms_modelInfo[i].m_filename);
#endif

        if (NULL != pModel)
        {

            Nv::NvModelExtVK* pModelVK = m_models[i] =
                Nv::NvModelExtVK::Create(vk(), pModel);
            if (NULL == pModelVK)
            {
                continue;
            }

            // Either Assimp or our art is completely broken, so bone
            // transforms are garbage. 
            // Use the static shader and ignore skin until we can untangle it
            ModelDesc& desc = ms_modelInfo[i];
            desc.m_center = Rotate(desc.m_fixupXfm, pModelVK->GetCenter());
            desc.m_halfExtents = Rotate(desc.m_fixupXfm, pModelVK->GetMaxExt() - ms_modelInfo[i].m_center);
        }
    }

#if defined(EMBEDDED_RESOURCES)

    // Load the skybox
    if (!vk().uploadTextureFromDDSData((const char*)sand_dds, sand_dds_len, m_skyboxSandTex))
    {
        LOGE("ThreadedRenderingVk: Error - Unable to load textures/sand.dds")
    }

    if (!vk().uploadTextureFromDDSData((const char*)Gradient_dds, Gradient_dds_len, m_skyboxGradientTex))
    {
        LOGE("ThreadedRenderingVk: Error - Unable to load textures/Gradient.dds")
    }

    if (!vk().uploadTextureFromDDSData((const char*)caustic1_dds, caustic1_dds_len, m_caustic1Tex))
    {
        LOGE("ThreadedRenderingVk: Error - Unable to load textures/Caustic1.dds")
    }

    if (!vk().uploadTextureFromDDSData((const char*)caustic2_dds, caustic2_dds_len, m_caustic2Tex))
    {
        LOGE("ThreadedRenderingVk: Error - Unable to load textures/Caustic2.dds")
    }


#else
    // Load the skybox
    if (!vk().uploadTextureFromDDSFile("textures/sand.dds", m_skyboxSandTex))
    {
        LOGE("ThreadedRenderingVk: Error - Unable to load textures/sand.dds")
    }

    if (!vk().uploadTextureFromDDSFile("textures/Gradient.dds", m_skyboxGradientTex))
    {
        LOGE("ThreadedRenderingVk: Error - Unable to load textures/Gradient.dds")
    }

    if (!vk().uploadTextureFromDDSFile("textures/caustic1.dds", m_caustic1Tex))
    {
        LOGE("ThreadedRenderingVk: Error - Unable to load textures/Caustic1.dds")
    }

    if (!vk().uploadTextureFromDDSFile("textures/caustic2.dds", m_caustic2Tex))
    {
        LOGE("ThreadedRenderingVk: Error - Unable to load textures/Caustic2.dds")
    }
#endif
    // Assign some values which apply to the entire scene and update once per frame.
    m_lightingUBO.Initialize(vk());
    m_lightingUBO->m_lightPosition = nv::vec4f(1.0f, 1.0f, 1.0f, 0.0f);
    m_lightingUBO->m_lightAmbient = nv::vec4f(0.4f, 0.4f, 0.4f, 1.0f);
    m_lightingUBO->m_lightDiffuse = nv::vec4f(0.7f, 0.7f, 0.7f, 1.0f);
    m_lightingUBO->m_causticOffset = m_currentTime * m_causticSpeed;
    m_lightingUBO->m_causticTiling = m_causticTiling;
    m_lightingUBO.Update();

    m_projUBO.Initialize(vk());

    VkPipelineDepthStencilStateCreateInfo depthTestAndWrite;

    depthTestAndWrite.sType =  VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO ;
    depthTestAndWrite.depthTestEnable = VK_TRUE;
    depthTestAndWrite.depthWriteEnable = VK_TRUE;
    depthTestAndWrite.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthTestAndWrite.depthBoundsTestEnable = VK_FALSE;
    depthTestAndWrite.stencilTestEnable = VK_FALSE;
    depthTestAndWrite.minDepthBounds = 0.0f;
    depthTestAndWrite.maxDepthBounds = 1.0f;

    VkPipelineDepthStencilStateCreateInfo depthTestNoWrite;
    depthTestNoWrite.sType =  VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO ;
    depthTestNoWrite.depthTestEnable = VK_TRUE;
    depthTestNoWrite.depthWriteEnable = VK_FALSE;
    depthTestNoWrite.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthTestNoWrite.depthBoundsTestEnable = VK_FALSE;
    depthTestNoWrite.stencilTestEnable = VK_FALSE;
    depthTestNoWrite.minDepthBounds = 0.0f;
    depthTestNoWrite.maxDepthBounds = 1.0f;

    VkPipelineColorBlendAttachmentState colorStateBlend = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorStateBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorStateBlend.blendEnable = VK_TRUE;
    colorStateBlend.alphaBlendOp = VK_BLEND_OP_ADD;
    colorStateBlend.colorBlendOp = VK_BLEND_OP_ADD;
    colorStateBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorStateBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorStateBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorStateBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    VkPipelineColorBlendStateCreateInfo colorInfoBlend = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorInfoBlend.logicOpEnable = VK_FALSE;
    colorInfoBlend.attachmentCount = 1;
    colorInfoBlend.pAttachments = &colorStateBlend;
    colorInfoBlend.blendConstants[0] = 1.0f;
    colorInfoBlend.blendConstants[1] = 1.0f;
    colorInfoBlend.blendConstants[2] = 1.0f;
    colorInfoBlend.blendConstants[3] = 1.0f;

    VkPipelineColorBlendAttachmentState colorStateNoBlend = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorStateNoBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorStateNoBlend.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorInfoNoBlend = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorInfoNoBlend.logicOpEnable = VK_FALSE;
    colorInfoNoBlend.attachmentCount = 1;
    colorInfoNoBlend.pAttachments = &colorStateNoBlend;

    m_quad = NvQuadVK::Create(vk());

    VkDescriptorSetLayoutBinding binding[DESC_COUNT];
    for (uint32_t i = 0; i < DESC_COUNT; i++) {
        binding[i].binding = i;
        binding[i].descriptorCount = 1;
        binding[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        binding[i].descriptorType = i < DESC_FIRST_TEX
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
            : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding[i].pImmutableSamplers = NULL;
    }

    {
        VkDescriptorSetLayoutCreateInfo descriptorSetEntry = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        descriptorSetEntry.bindingCount = DESC_COUNT;
        descriptorSetEntry.pBindings = binding;

        result = vkCreateDescriptorSetLayout(device(), &descriptorSetEntry, 0, mDescriptorSetLayout);
        CHECK_VK_RESULT();
    }

    // Create descriptor region and set
    VkDescriptorPoolSize descriptorPoolInfo[2];

    descriptorPoolInfo[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    descriptorPoolInfo[0].descriptorCount = DESC_COUNT;
    descriptorPoolInfo[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorPoolInfo[1].descriptorCount = DESC_COUNT;

    VkDescriptorPoolCreateInfo descriptorRegionInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descriptorRegionInfo.maxSets = DESC_COUNT;
    descriptorRegionInfo.poolSizeCount = 2;
    descriptorRegionInfo.pPoolSizes = descriptorPoolInfo;
    VkDescriptorPool descriptorRegion;
    result = vkCreateDescriptorPool(device(), &descriptorRegionInfo, NULL, &descriptorRegion);
    CHECK_VK_RESULT();

    {
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        descriptorSetAllocateInfo.descriptorPool = descriptorRegion;
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts = mDescriptorSetLayout;
        result = vkAllocateDescriptorSets(device(), &descriptorSetAllocateInfo, &mDescriptorSet);
        CHECK_VK_RESULT();
    }

    VkWriteDescriptorSet writeDescriptorSets[DESC_COUNT];
    memset(writeDescriptorSets, 0, sizeof(writeDescriptorSets));

    for (uint32_t i = 0; i < DESC_COUNT; i++) {
        writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[i].dstSet = mDescriptorSet;
        writeDescriptorSets[i].dstBinding = i;
        writeDescriptorSets[i].dstArrayElement = 0;
        writeDescriptorSets[i].descriptorCount = 1;
        writeDescriptorSets[i].descriptorType = i < DESC_FIRST_TEX
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
            : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }

    {
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = mDescriptorSetLayout;
        result = vkCreatePipelineLayout(device(), &pipelineLayoutCreateInfo, 0, &m_pipeLayout);
        CHECK_VK_RESULT();
    }

    VkDescriptorSetLayoutBinding fishBinding[3];
    fishBinding[0].binding = 0;
    fishBinding[0].descriptorCount = 1;
    fishBinding[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    fishBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    fishBinding[0].pImmutableSamplers = NULL;
    fishBinding[1].binding = 1;
    fishBinding[1].descriptorCount = 1;
    fishBinding[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    fishBinding[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    fishBinding[1].pImmutableSamplers = NULL;
    fishBinding[2].binding = 2;
    fishBinding[2].descriptorCount = 1;
    fishBinding[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    fishBinding[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    fishBinding[2].pImmutableSamplers = NULL;

    {
        VkDescriptorSetLayoutCreateInfo descriptorSetEntry = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        descriptorSetEntry.bindingCount = 3;
        descriptorSetEntry.pBindings = fishBinding;

        result = vkCreateDescriptorSetLayout(device(), &descriptorSetEntry, 0, &mDescriptorSetLayout[1]);
        CHECK_VK_RESULT();
    }

    {
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutCreateInfo.setLayoutCount = 2;
        pipelineLayoutCreateInfo.pSetLayouts = mDescriptorSetLayout;
        result = vkCreatePipelineLayout(device(), &pipelineLayoutCreateInfo, 0, &m_fishPipeLayout);
        CHECK_VK_RESULT();
    }

    {
        // Create fish descriptor region and set
        VkDescriptorPoolSize fishDescriptorPoolInfo[2];

        fishDescriptorPoolInfo[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        fishDescriptorPoolInfo[0].descriptorCount = MAX_SCHOOL_COUNT;
        fishDescriptorPoolInfo[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        fishDescriptorPoolInfo[1].descriptorCount = 2 * MAX_SCHOOL_COUNT;

        VkDescriptorPoolCreateInfo descriptorRegionInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        descriptorRegionInfo.maxSets = 3 * MAX_SCHOOL_COUNT;
        descriptorRegionInfo.poolSizeCount = 2;
        descriptorRegionInfo.pPoolSizes = fishDescriptorPoolInfo;
        result = vkCreateDescriptorPool(device(), &descriptorRegionInfo, NULL, &mDescriptorRegion);
        CHECK_VK_RESULT();
    }

    m_fishAlloc.Initialize(vk(), MAX_SCHOOL_COUNT * (
        NvSimpleUBO<School::SchoolUBO>::GetBufferSpace() +
        NvSimpleUBO<nv::matrix4f>::GetBufferSpace()));

    m_vboAlloc.Initialize(vk(), 4 * MAX_SCHOOL_COUNT * MAX_INSTANCE_COUNT *
        sizeof(School::FishInstanceData));

    SetNumSchools(m_initSchools);
    vkQueueWaitIdle(queue());

    VkDescriptorBufferInfo projDesc;
    m_projUBO.GetDesc(projDesc);
    writeDescriptorSets[DESC_PROJ_UBO].pBufferInfo = &projDesc;

    VkDescriptorBufferInfo lightDesc;
    m_lightingUBO.GetDesc(lightDesc);
    writeDescriptorSets[DESC_LIGHT_UBO].pBufferInfo = &lightDesc;

    VkDescriptorImageInfo caus1Desc = {};
    caus1Desc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    caus1Desc.imageView = m_caustic1Tex.view;
    caus1Desc.sampler = m_wrapSampler;
    writeDescriptorSets[DESC_CAUS1_TEX].pImageInfo = &caus1Desc;

    VkDescriptorImageInfo caus2Desc = {};
    caus2Desc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    caus2Desc.imageView = m_caustic2Tex.view;
    caus2Desc.sampler = m_wrapSampler;
    writeDescriptorSets[DESC_CAUS2_TEX].pImageInfo = &caus2Desc;

    VkDescriptorImageInfo sandDesc = {};
    sandDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sandDesc.imageView = m_skyboxSandTex.view;
    sandDesc.sampler = m_wrapSampler;
    writeDescriptorSets[DESC_SAND_TEX].pImageInfo = &sandDesc;

    VkDescriptorImageInfo gradDesc = {};
    gradDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    gradDesc.imageView = m_skyboxGradientTex.view;
    gradDesc.sampler = m_linearSampler;
    writeDescriptorSets[DESC_GRAD_TEX].pImageInfo = &gradDesc;

    vkUpdateDescriptorSets(device(), DESC_COUNT, writeDescriptorSets, 0, 0);

    InitPipeline(2, fishShaderStages, 
        &m_schools[0]->getModel()->GetModel()->GetMesh(0)->getVIInfo(),
        &m_schools[0]->getModel()->GetModel()->GetMesh(0)->getIAInfo(),
        &colorInfoBlend, &depthTestAndWrite, m_fishPipeLayout, &m_fishPipe);

    InitPipeline(2, groundShaderStages, &m_quad->getVIInfo(), &m_quad->getIAInfo(),
        &colorInfoNoBlend, &depthTestNoWrite, m_pipeLayout, &m_groundPlanePipe);

    InitPipeline(2, skyShaderStages, &m_quad->getVIInfo(), &m_quad->getIAInfo(),
        &colorInfoNoBlend, &depthTestNoWrite, m_pipeLayout, &m_skyboxColorPipe);

    ResetFish(false);

    InitThreads();

    if (isTestMode()) {
        for (uint32_t j = 0; j < 120; j++) {

            for (uint32_t i = 0; i < m_activeSchools; i++) {
                m_schools[i]->SetInstanceCount(m_uiInstanceCount);
                m_schools[i]->Animate(1.0f / 60.0f, &m_schoolStateMgr, m_avoidance);
                m_schools[i]->Update();
            }
        }
        m_animPaused = true;
    }
    m_uiSchoolInfoId = 0;
    SchoolDesc& desc = m_schoolDescs[m_uiSchoolInfoId];
    m_uiSchoolDisplayModelIndex = desc.m_modelId;
}

void ThreadedRenderingVk::shutdownRendering(void) {
    if (getAppContext() && device())
        vkDeviceWaitIdle(device());

    CleanThreads();
    CleanRendering();
}

void ThreadedRenderingVk::InitPipeline(uint32_t shaderCount, 
    VkPipelineShaderStageCreateInfo* shaderStages,
    VkPipelineVertexInputStateCreateInfo* pVertexInputState,
    VkPipelineInputAssemblyStateCreateInfo* pInputAssemblyState,
    VkPipelineColorBlendStateCreateInfo* pBlendState,
    VkPipelineDepthStencilStateCreateInfo* pDSState,
    VkPipelineLayout& layout,
    VkPipeline* pipeline)
{
    VkResult result;
    // Create static state info for the mPipeline.

    // set dynamically
    VkPipelineViewportStateCreateInfo vpStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vpStateInfo.pNext = 0;
    vpStateInfo.viewportCount = 1;
    vpStateInfo.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rsStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rsStateInfo.depthClampEnable = VK_FALSE;
    rsStateInfo.rasterizerDiscardEnable = VK_FALSE;
    rsStateInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rsStateInfo.cullMode = VK_CULL_MODE_NONE;
    rsStateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsStateInfo.lineWidth = 1.0f;

    /*VkPipelineColorBlendAttachmentState attachments[1] = {};
    attachments[0].blendEnable = VK_FALSE;
    attachments[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cbStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cbStateInfo.logicOpEnable = VK_FALSE;
    cbStateInfo.attachmentCount = ARRAY_SIZE(attachments);
    cbStateInfo.pAttachments = attachments;*/

    VkPipelineMultisampleStateCreateInfo msStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    msStateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    msStateInfo.alphaToCoverageEnable = VK_FALSE;
    msStateInfo.sampleShadingEnable = VK_FALSE;
    msStateInfo.minSampleShading = 1.0f;
    uint32_t smplMask = 0x1;
    msStateInfo.pSampleMask = &smplMask;

    VkPipelineTessellationStateCreateInfo tessStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO };
    tessStateInfo.patchControlPoints = 0;

    VkPipelineDynamicStateCreateInfo dynStateInfo;
    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    memset(&dynStateInfo, 0, sizeof(dynStateInfo));
    dynStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynStateInfo.dynamicStateCount = 2;
    dynStateInfo.pDynamicStates = dynStates;

    // Shaders
    VkGraphicsPipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    // Assuming that all sub-meshes in an ModelExt have the same layout...
    pipelineInfo.pVertexInputState = pVertexInputState;
    pipelineInfo.pInputAssemblyState = pInputAssemblyState;
    pipelineInfo.pViewportState = &vpStateInfo;
    pipelineInfo.pRasterizationState = &rsStateInfo;
    pipelineInfo.pColorBlendState = pBlendState;
    pipelineInfo.pDepthStencilState = pDSState;
    pipelineInfo.pMultisampleState = &msStateInfo;
    pipelineInfo.pTessellationState = &tessStateInfo;
    pipelineInfo.pDynamicState = &dynStateInfo;

    pipelineInfo.stageCount = shaderCount;
    pipelineInfo.pStages = shaderStages;

    pipelineInfo.renderPass = vk().mainRenderTarget()->clearRenderPass();
    pipelineInfo.subpass = 0;

    pipelineInfo.layout = layout;

    result = vkCreateGraphicsPipelines(device(), VK_NULL_HANDLE, 1, &pipelineInfo, NULL,
        pipeline);
    CHECK_VK_RESULT();
}

void ThreadedRenderingVk::InitializeSamplers()
{
    VkSamplerCreateInfo samplerCreateInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.mipLodBias = 0.0;
    samplerCreateInfo.maxAnisotropy = 1;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerCreateInfo.minLod = 0.0;
    samplerCreateInfo.maxLod = 16.0;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

    vkCreateSampler(device(), &samplerCreateInfo, 0, &m_linearSampler);

    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    vkCreateSampler(device(), &samplerCreateInfo, 0, &m_nearestSampler);

    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    vkCreateSampler(device(), &samplerCreateInfo, 0, &m_wrapSampler);
}

uint32_t ThreadedRenderingVk::SetNumSchools(uint32_t numSchools)
{
    VkResult result;

    if (numSchools > m_maxSchools)
    {
        numSchools = m_maxSchools;
    }
    m_activeSchools = numSchools;

    if (m_schoolDescs.size() < m_activeSchools)
    {
        InitializeSchoolDescriptions(m_activeSchools);
    }

    nv::vec3f location(0.0f, 0.0f, 0.0f);

    if (m_schools.size() < m_activeSchools)
    {
        // We need to increase the size of our array of schools and initialize the new ones
        uint32_t schoolIndex = m_schools.size();
        m_schools.resize(m_activeSchools);

        int32_t newSchools = m_activeSchools - schoolIndex;

        //VkDescriptorSet* descSets = NULL;
        if (newSchools > 0) {
            for (; schoolIndex < m_schools.size(); ++schoolIndex)
            {
                SchoolDesc& desc = m_schoolDescs[schoolIndex];
                School* pSchool = new School(desc.m_flocking);
                if (m_uiResetMode == RESET_RANDOM)
                {
                    nv::vec3f tankSize = ms_tankMax - ms_tankMin;
                    location = nv::vec3f((float)rand() / (float)RAND_MAX * tankSize.x,
                        (float)rand() / (float)RAND_MAX * tankSize.y,
                        (float)rand() / (float)RAND_MAX * tankSize.z);
                    location += ms_tankMin;
                }
                // Account for scaling in the transforms and extents that we are providing
                // to the school
                nv::matrix4f finalTransform;
                finalTransform.set_scale(desc.m_scale);
                finalTransform *= ms_modelInfo[desc.m_modelId].GetCenteringTransform();

                VkDescriptorSet descSet;
                {
                    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
                    descriptorSetAllocateInfo.descriptorPool = mDescriptorRegion;
                    descriptorSetAllocateInfo.descriptorSetCount = 1;
                    descriptorSetAllocateInfo.pSetLayouts = &mDescriptorSetLayout[1];
                    result = vkAllocateDescriptorSets(device(), &descriptorSetAllocateInfo, &descSet);
                    CHECK_VK_RESULT();
                }

                if (!pSchool->Initialize(vk(),
                    schoolIndex,
                    m_models[desc.m_modelId],
                    ms_modelInfo[desc.m_modelId].m_tailStartZ * desc.m_scale,
                    finalTransform,
                    ms_modelInfo[desc.m_modelId].m_halfExtents * desc.m_scale,
                    desc.m_numFish,
                    MAX_INSTANCE_COUNT,
                    location,
                    descSet,
                    m_fishPipeLayout,
                    m_fishAlloc,
                    m_vboAlloc)) {
                    return 0;
                }

                m_schools[schoolIndex] = pSchool;
            }
        }
    }

    // Update our readout of total number of active fish
    m_uiFishCount = m_activeSchools * m_uiInstanceCount;
    return m_activeSchools;
}

void ThreadedRenderingVk::InitializeSchoolDescriptions(uint32_t numSchools)
{
    uint32_t schoolIndex = m_schoolDescs.size();
    m_schoolDescs.resize(numSchools);
    for (; schoolIndex < numSchools; ++schoolIndex)
    {
        SchoolDesc& desc = m_schoolDescs[schoolIndex];
        desc = ms_schoolInfo[schoolIndex % MODEL_COUNT];
        desc.m_numFish = INSTANCE_COUNT;
    }
}

uint32_t ThreadedRenderingVk::SetThreadNum(uint32_t numThreads)
{
    if (MAX_THREAD_COUNT < numThreads)
    {
        numThreads = MAX_THREAD_COUNT;
    }
    m_requestedActiveThreads = numThreads;

    return m_requestedActiveThreads;
}

void ThreadedRenderingVk::ResetFish(bool fishsplosion) {
    if (fishsplosion) {
        nv::vec3f location = (ms_tankMin + ms_tankMax) * 0.5f;
        for (uint32_t schoolIndex = 0; schoolIndex < m_schools.size(); ++schoolIndex)
        {
            School* pSchool = m_schools[schoolIndex];
            pSchool->ResetToLocation(location);
        }
    }
    else {
        for (uint32_t schoolIndex = 0; schoolIndex < m_schools.size(); ++schoolIndex)
        {
            School* pSchool = m_schools[schoolIndex];
            nv::vec3f tankSize = ms_tankMax - ms_tankMin;
            nv::vec3f location((float)rand() / (float)RAND_MAX * tankSize.x,
                (float)rand() / (float)RAND_MAX * tankSize.y,
                (float)rand() / (float)RAND_MAX * tankSize.z);
            location += ms_tankMin;
            pSchool->ResetToLocation(location);
        }
    }
}

void ThreadedRenderingVk::reshape(int32_t width, int32_t height)
{
    //setting the perspective projection matrix
    nv::perspective(m_projUBO_Data.m_projectionMatrix, NV_PI / 3.0f,
        static_cast<float>(NvSampleApp::m_width) /
        static_cast<float>(NvSampleApp::m_height),
        0.1f, 100.0f);

    //setting the inverse perspective projection matrix
    m_projUBO_Data.m_inverseProjMatrix =
        nv::inverse(m_projUBO_Data.m_projectionMatrix);

    //setting the perspective projection matrix
    nv::perspectiveVk(m_projUBO->m_projectionMatrix, NV_PI / 3.0f,
        static_cast<float>(NvSampleApp::m_width) /
        static_cast<float>(NvSampleApp::m_height),
        0.1f, 100.0f);

    //setting the inverse perspective projection matrix
    m_projUBO->m_inverseProjMatrix =
        nv::inverse(m_projUBO->m_projectionMatrix);
}


float ThreadedRenderingVk::getClampedFrameTime() {
    float delta = getFrameDeltaTime();
    if (delta > 0.2f)
        delta = 0.2f;
    return delta;
}

static bool refresh = false;

void ThreadedRenderingVk::draw(void)
{
    if (!refresh)
    {
        reshape(1920,1080);
        ResetFish(true);
        refresh = true;
    }

    // Update the active number of threads
    if (m_requestedActiveThreads != m_activeThreads)
    {
        m_activeThreads = m_requestedActiveThreads;
        vkDeviceWaitIdle(device());
    }
    if (m_requestedThreadedRendering != m_threadedRendering)
    {
        m_threadedRendering = m_requestedThreadedRendering;
        vkDeviceWaitIdle(device());
    }

    if (m_flushPerFrame)
        vkDeviceWaitIdle(device());
    VkResult result;

    neighborOffset = (neighborOffset + 1) % (6 - neighborSkip);

    m_currentTime += getClampedFrameTime();

    //printf("time:%f\n", m_currentTime);

    m_schoolStateMgr.BeginFrame(m_activeSchools);
#if FISH_DEBUG
    LOGI("\n################################################################");
    LOGI("BEGINNING OF FRAME");
#endif

    // Get the current view matrix (according to user input through mouse,
    // gamepad, etc.)
    m_projUBO->m_viewMatrix = nv::matrix4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -15, -40, 1) ;
    m_projUBO->m_inverseViewMatrix = nv::matrix4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 15, 40, 1);
    m_projUBO.Update();

    m_subCommandBuffers.BeginFrame();
    m_lightingUBO->m_causticOffset = m_currentTime * m_causticSpeed;
    m_lightingUBO->m_causticTiling = m_causticTiling;
    m_lightingUBO.Update();

    // Render the requested content (from dropdown menu in TweakBar UI)


    // BEGIN NEW RENDERING CODE

    // The SimpleCommandBuffer class overloads the * operator
    VkCommandBuffer cmd = vk().getMainCommandBuffer();

    VkRenderPassBeginInfo renderPassBeginInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };

    renderPassBeginInfo.renderPass = vk().mainRenderTarget()->clearRenderPass();
    renderPassBeginInfo.framebuffer = vk().mainRenderTarget()->frameBuffer();
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = m_width;
    renderPassBeginInfo.renderArea.extent.height = m_height;

    VkClearValue clearValues[2];
    clearValues[0].color.float32[0] = 0.33f;
    clearValues[0].color.float32[1] = 0.44f;
    clearValues[0].color.float32[2] = 0.66f;
    clearValues[0].color.float32[3] = 1.0f;
    clearValues[1].depthStencil.depth = 1.0f;
    clearValues[1].depthStencil.stencil = 0;


    renderPassBeginInfo.pClearValues = clearValues;
    renderPassBeginInfo.clearValueCount = 2;

    vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

    VkCommandBufferInheritanceInfo inherit = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO };
    inherit.framebuffer = vk().mainRenderTarget()->frameBuffer();
    inherit.renderPass = vk().mainRenderTarget()->clearRenderPass();

    VkCommandBuffer secCmd = m_subCommandBuffers[0];

    VkCommandBufferBeginInfo secBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    secBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT |
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
        VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    secBeginInfo.pInheritanceInfo = &inherit;
    result = vkBeginCommandBuffer(secCmd, &secBeginInfo);

    CHECK_VK_RESULT();
    {
        VkViewport vp;
        VkRect2D sc;
        vp.x = 0;
        vp.y = 0;
        vp.height = (float)(m_height);
        vp.width = (float)(m_width);
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;

        sc.offset.x = 0;
        sc.offset.y = 0;
        sc.extent.width = vp.width;
        sc.extent.height = vp.height;

        vkCmdSetViewport(secCmd, 0, 1, &vp);
        vkCmdSetScissor(secCmd, 0, 1, &sc);
    }
    {
        uint32_t offsets[DESC_COUNT] = { 0 };
        offsets[0] = m_projUBO.getDynamicOffset();
        offsets[1] = m_lightingUBO.getDynamicOffset();
        offsets[2] = m_schools[0]->getModelUBO().getDynamicOffset();
        offsets[3] = m_schools[0]->getMeshUBO().getDynamicOffset();
        vkCmdBindDescriptorSets(secCmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeLayout, 0, 1, &mDescriptorSet, DESC_FIRST_TEX, offsets);

        DrawSkyboxColorDepth(secCmd);
        DrawGroundPlane(secCmd);
    }

    result = vkEndCommandBuffer(secCmd);
    CHECK_VK_RESULT();

    vkCmdExecuteCommands(cmd, 1, &secCmd);

    {
        //CPU_TIMER_SCOPE(CPU_TIMER_MAIN_WAIT);

        for (uint32_t i = 0; i < MAX_THREAD_COUNT; i++) {
            ThreadData& thread = m_Threads[i];
            thread.m_cmdBufferOpen = false;
            thread.m_drawCallCount = 0;
        }

        uint32_t schoolsPerThread = m_activeSchools / m_activeThreads;
        uint32_t remainderSchools = m_activeSchools % m_activeThreads;
        uint32_t baseSchool = 0;
        for (uint32_t i = 0; i < m_activeThreads; i++) {
            ThreadData& thread = m_Threads[i];
            thread.m_baseSchoolIndex = baseSchool;
            thread.m_schoolCount = schoolsPerThread;
            // distribute the remainder evenly
            if (remainderSchools > 0) {
                thread.m_schoolCount++;
                remainderSchools--;
            }
                
            baseSchool += thread.m_schoolCount;
        }

        m_doneCount = 0;
        m_drawCallCount = 0;

        m_FrameStartLock->lockMutex();
        {
                        m_frameID++;
            m_FrameStartCV->broadcastConditionVariable();
        }
        m_FrameStartLock->unlockMutex();

        m_DoneCountLock->lockMutex();
        {
            while (m_doneCount != m_activeSchools)
            {
                m_DoneCountCV->waitConditionVariable(m_DoneCountLock);
            }
        }
        m_DoneCountLock->unlockMutex();
    }

    {
        //CPU_TIMER_SCOPE(CPU_TIMER_MAIN_CMD_BUILD);

        if (/*m_threadedRendering*/false) {
            for (uint32_t i = 0; i < m_activeThreads; i++) {
                VkCommandBuffer secCmd = m_subCommandBuffers[i + 1];
                vkCmdExecuteCommands(cmd, 1, &secCmd);
                m_Threads[i].m_cmdBufferOpen = false;
                m_drawCallCount += m_Threads[i].m_drawCallCount;
            }
        }
        else
        {
            // round robin through the available buffers, so they are all used
            // over a few frames; this tries to match the idea that the threading
            // code will also hit all of these buffers when enabled
            int bufIndex = (m_frameLogicalClock % MAX_THREAD_COUNT) + 1;
            VkCommandBuffer secCmd = m_subCommandBuffers[bufIndex];

            uint32_t schoolIndex = 0;
            for (; schoolIndex < m_activeSchools; ++schoolIndex)
            {
                School* pSchool = m_schools[schoolIndex];
                m_drawCallCount += DrawSchool(secCmd, pSchool, schoolIndex == 0);
            }
            result = vkEndCommandBuffer(secCmd);
            CHECK_VK_RESULT();
            vkCmdExecuteCommands(cmd, 1, &secCmd);
        }
    }

    vkCmdEndRenderPass(cmd);

    vk().submitMainCommandBuffer();

    m_subCommandBuffers.DoneWithFrame();

#if FISH_DEBUG
    LOGI("END OF FRAME");
    LOGI("################################################################\n");
#endif
    m_frameLogicalClock++;
}

//-----------------------------------------------------------------------------
// PRIVATE METHODS

void ThreadedRenderingVk::CleanRendering(void)
{
}

namespace nvidia{
NV_FOUNDATION_API NvAssertHandler& NvGetAssertHandler()
{
    static NvAssertHandler obj;
    return obj;
}
}

void ThreadedRenderingVk::InitThreads(void)
{
    //ASSERT(nullptr != pDevice);

    NvThreadManager* threadManager = getThreadManagerInstance();
    NV_ASSERT(NULL != threadManager);

    NV_ASSERT(m_FrameStartLock == NULL);
    NV_ASSERT(m_FrameStartCV == NULL);
    NV_ASSERT(m_NeedsUpdateQueueLock == NULL);
    NV_ASSERT(m_DoneCountLock == NULL);
    NV_ASSERT(m_DoneCountCV == NULL);

    m_FrameStartLock =
        threadManager->initializeMutex(false, NvMutex::MutexLockLevelInitial);
    m_FrameStartCV =
        threadManager->initializeConditionVariable();
    m_NeedsUpdateQueueLock =
        threadManager->initializeMutex(false, NvMutex::MutexLockLevelInitial);
    m_DoneCountLock =
        threadManager->initializeMutex(false, NvMutex::MutexLockLevelInitial);
    m_DoneCountCV = threadManager->initializeConditionVariable();

    NV_ASSERT(m_FrameStartLock != NULL);
    NV_ASSERT(m_FrameStartCV != NULL);
    NV_ASSERT(m_NeedsUpdateQueueLock != NULL);
    NV_ASSERT(m_DoneCountLock != NULL);
    NV_ASSERT(m_DoneCountCV != NULL);

    m_subCommandBuffers.Initialize(&vk(), m_queueMutexPtr, false);

    for (intptr_t i = 0; i < MAX_THREAD_COUNT; i++)
    {
        ThreadData& thread = m_Threads[i];
        if (thread.m_thread != NULL)
            delete thread.m_thread;

        //void* threadIndex = reinterpret_cast<void*>(i);
        m_Threads[i].m_thread =
            threadManager->createThread(ThreadJobFunctionThunk, &thread,
                                        &(m_ThreadStacks[i]),
                                        THREAD_STACK_SIZE,
                                        NvThread::DefaultThreadPriority);

        NV_ASSERT(thread.m_thread != NULL);

        thread.m_thread->startThread();
    }
}

void ThreadedRenderingVk::CleanThreads(void)
{
    //NV_ASSERT(nullptr != pDevice);

    NvThreadManager* threadManager = getThreadManagerInstance();
    NV_ASSERT(NULL != threadManager);

    m_running = false;
    if (m_FrameStartCV)
        m_FrameStartCV->broadcastConditionVariable();

    for (uint32_t i = 0; i < MAX_THREAD_COUNT; i++)
    {
        if (m_Threads[i].m_thread != NULL)
        {
            m_Threads[i].m_thread->waitThread();
            threadManager->destroyThread(m_Threads[i].m_thread);
            m_Threads[i].m_thread = NULL;
        }
    }

    if (m_NeedsUpdateQueueLock)
        threadManager->finalizeMutex(m_NeedsUpdateQueueLock);
    if (m_DoneCountLock)
        threadManager->finalizeMutex(m_DoneCountLock);
    if (m_DoneCountCV)
        threadManager->finalizeConditionVariable(m_DoneCountCV);

    m_FrameStartLock = NULL;
    m_FrameStartCV = NULL;
    m_NeedsUpdateQueueLock = NULL;
    m_DoneCountLock = NULL;
    m_DoneCountCV = NULL;
}

uint32_t ThreadedRenderingVk::DrawSchool(VkCommandBuffer& cmd, School* pSchool, bool openBuffer) {
    if (openBuffer) {
        VkCommandBufferInheritanceInfo inherit = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO };
        inherit.framebuffer = vk().mainRenderTarget()->frameBuffer();
        inherit.renderPass = vk().mainRenderTarget()->clearRenderPass();

        VkCommandBufferBeginInfo secBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        secBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT |
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
            VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        secBeginInfo.pInheritanceInfo = &inherit;
        vkBeginCommandBuffer(cmd, &secBeginInfo);
        {
            VkViewport vp;
            VkRect2D sc;
            vp.x = 0;
            vp.y = 0;
            vp.height = (float)(m_height);
            vp.width = (float)(m_width);
            vp.minDepth = 0.0f;
            vp.maxDepth = 1.0f;

            sc.offset.x = 0;
            sc.offset.y = 0;
            sc.extent.width = vp.width;
            sc.extent.height = vp.height;

            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fishPipe);

        uint32_t offsets[DESC_COUNT] = { 0 };
        offsets[0] = m_projUBO.getDynamicOffset();
        offsets[1] = m_lightingUBO.getDynamicOffset();
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fishPipeLayout, 0, 1, &mDescriptorSet, DESC_FIRST_TEX, offsets);
        //printf("DrawSchool:%d,%d\n",m_width, m_height);
    }
    return pSchool->Render(cmd, m_uiBatchSizeActual);
}

//  Draws the skybox with lighting in color and depth
void ThreadedRenderingVk::DrawSkyboxColorDepth(VkCommandBuffer& cmd)
{
    VkViewport vp;
    VkRect2D sc;
    vp.x = 0;
    vp.y = 0;
    vp.height = (float)(m_height);
    vp.width = (float)(m_width);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    sc.offset.x = 0;
    sc.offset.y = 0;
    sc.extent.width = vp.width;
    sc.extent.height = vp.height;

    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyboxColorPipe);
    {
        m_quad->Draw(cmd);
    }
}

void ThreadedRenderingVk::DrawGroundPlane(VkCommandBuffer& cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_groundPlanePipe);
    {
        m_quad->Draw(cmd);
    }
}

void ThreadedRenderingVk::UpdateSchool(uint32_t threadIndex,
    uint32_t schoolIndex, School* pSchool)
{
    pSchool->SetInstanceCount(m_uiInstanceCount);
    {
        pSchool->Animate(getClampedFrameTime() * 0.1, &m_schoolStateMgr, m_avoidance);
    }

    pSchool->Update();
}

// Static Data to define available models
static const nv::matrix4f sc_yawNeg90(
     0.0f,  0.0f,  1.0f,  0.0f,
     0.0f,  1.0f,  0.0f,  0.0f,
    -1.0f,  0.0f,  0.0f,  0.0f,
     0.0f,  0.0f,  0.0f,  1.0f
);

static const nv::matrix4f sc_yaw180(
    -1.0f,  0.0f,  0.0f,  0.0f,
     0.0f,  1.0f,  0.0f,  0.0f,
     0.0f,  0.0f, -1.0f,  0.0f,
     0.0f,  0.0f,  0.0f,  1.0f
);

static const nv::vec3f sc_zeroVec(0.0f, 0.0f, 0.0f);


#if defined(EMBEDDED_RESOURCES)

#include "assets/models/Black_White_Fish_nve.h"
#include "assets/models/Blue_Fish_06_nve.h"
#include "assets/models/Cyan_Fish_nve.h"
#include "assets/models/Yellow_Fish_03_nve.h"
#include "assets/models/Yellow_Fish_08_nve.h"
#include "assets/models/Blue_Fish_02_nve.h"
#include "assets/models/Blue_Fish_07_nve.h"
#include "assets/models/Pink_Fish_nve.h"
#include "assets/models/Yellow_Fish_04_nve.h"
#include "assets/models/Yellow_Fish_09_nve.h"
#include "assets/models/Blue_Fish_03_nve.h"
#include "assets/models/Blue_Fish_08_nve.h"
#include "assets/models/Red_Fish_nve.h"
#include "assets/models/Yellow_Fish_05_nve.h"
#include "assets/models/Yellow_Fish_10_nve.h"
#include "assets/models/Blue_Fish_04_nve.h"
#include "assets/models/Blue_Fish_09_nve.h"
#include "assets/models/Violet_Fish_nve.h"
#include "assets/models/Yellow_Fish_06_nve.h"
#include "assets/models/Yellow_Fish_11_nve.h"
#include "assets/models/Blue_Fish_05_nve.h"
#include "assets/models/Blue_Fish_nve.h"
#include "assets/models/Yellow_Fish_02_nve.h"
#include "assets/models/Yellow_Fish_07_nve.h"
#include "assets/models/Yellow_Fish_nve.h"

#else

unsigned char *Black_White_Fish_nve = NULL;
unsigned char *Blue_Fish_nve = NULL;
unsigned char *Cyan_Fish_nve = NULL;
unsigned char *Yellow_Fish_03_nve = NULL;
unsigned char *Yellow_Fish_08_nve = NULL;
unsigned char *Blue_Fish_02_nve = NULL;
unsigned char *Blue_Fish_07_nve = NULL;
unsigned char *Pink_Fish_nve = NULL;
unsigned char *Yellow_Fish_04_nve = NULL;
unsigned char *Yellow_Fish_09_nve = NULL;
unsigned char *Blue_Fish_03_nve = NULL;
unsigned char *Blue_Fish_08_nve = NULL;
unsigned char *Red_Fish_nve = NULL;
unsigned char *Yellow_Fish_05_nve = NULL;
unsigned char *Yellow_Fish_10_nve = NULL;
unsigned char *Blue_Fish_04_nve = NULL;
unsigned char *Blue_Fish_09_nve = NULL;
unsigned char *Violet_Fish_nve = NULL;
unsigned char *Yellow_Fish_06_nve = NULL;
unsigned char *Yellow_Fish_11_nve = NULL;
unsigned char *Blue_Fish_05_nve = NULL;
unsigned char *Blue_Fish_nve = NULL;
unsigned char *Yellow_Fish_02_nve = NULL;
unsigned char *Yellow_Fish_07_nve = NULL;
unsigned char *Yellow_Fish_nve = NULL;

#endif

// Initialize the model desc array with data that we know, leaving the bounding
// box settings as zeroes.  We'll fill those in when the models are loaded.
ThreadedRenderingVk::ModelDesc ThreadedRenderingVk::ms_modelInfo[MODEL_COUNT] =
{
    { "Black & White Fish", "models/Black_White_Fish.nve", Black_White_Fish_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.10f },
    { "Blue Fish 1", "models/Blue_Fish.nve", Blue_Fish_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.25f },
    { "Blue Fish 2", "models/Blue_Fish_02.nve", Blue_Fish_02_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.30f },
    { "Blue Fish 3", "models/Blue_Fish_03.nve", Blue_Fish_03_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.25f },
    { "Blue Fish 4", "models/Blue_Fish_04.nve", Blue_Fish_04_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.30f },
    { "Blue Fish 5", "models/Blue_Fish_05.nve", Blue_Fish_05_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.25f },
    { "Blue Fish 6", "models/Blue_Fish_06.nve", Blue_Fish_06_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.15f },
    { "Blue Fish 7", "models/Blue_Fish_07.nve", Blue_Fish_07_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.35f },
    { "Blue Fish 8", "models/Blue_Fish_08.nve", Blue_Fish_08_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.25f },
    { "Blue Fish 9", "models/Blue_Fish_09.nve", Blue_Fish_09_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.20f },
    { "Cyan Fish", "models/Cyan_Fish.nve", Cyan_Fish_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.25f },
    { "Pink Fish", "models/Pink_Fish.nve", Pink_Fish_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.20f },
    { "Red Fish", "models/Red_Fish.nve",  Red_Fish_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.28f },
    { "Violet Fish", "models/Violet_Fish.nve", Violet_Fish_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.30f },
    { "Yellow Fish 1", "models/Yellow_Fish.nve", Yellow_Fish_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.40f },
    { "Yellow Fish 2", "models/Yellow_Fish_02.nve", Yellow_Fish_02_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.15f },
    { "Yellow Fish 3", "models/Yellow_Fish_03.nve", Yellow_Fish_03_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.25f },
    { "Yellow Fish 4", "models/Yellow_Fish_04.nve", Yellow_Fish_04_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.30f },
    { "Yellow Fish 5", "models/Yellow_Fish_05.nve", Yellow_Fish_05_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.25f },
    { "Yellow Fish 6", "models/Yellow_Fish_06.nve", Yellow_Fish_06_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.30f },
    { "Yellow Fish 7", "models/Yellow_Fish_07.nve", Yellow_Fish_07_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.25f },
    { "Yellow Fish 8", "models/Yellow_Fish_08.nve", Yellow_Fish_08_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.23f },
    { "Yellow Fish 9", "models/Yellow_Fish_09.nve", Yellow_Fish_09_nve, sc_yawNeg90, sc_zeroVec, sc_zeroVec, 0.25f },
    { "Yellow Fish 10", "models/Yellow_Fish_10.nve", Yellow_Fish_10_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.30f },
    { "Yellow Fish 11", "models/Yellow_Fish_11.nve", Yellow_Fish_11_nve, sc_yaw180, sc_zeroVec, sc_zeroVec, 0.32f }
};

SchoolFlockingParams ThreadedRenderingVk::ms_fishTypeDefs[FISHTYPE_COUNT] =
{
    //     |       |Goal|      Spawn Zone       |Neighbor|Spawn|          |<***************** Strengths ****************>|             
    // Spd |Inertia|Size|    Min         Max    |Distance|Range|Aggression| Goal |Alignment|Repulsion |Cohesion|Avoidance|
    SchoolFlockingParams(1.5f,  16.0f, 8.0f, ms_tankMin, ms_tankMax,  4.00f,  0.01f,   0.9f,    0.1f,   0.1f,    0.10f,      0.5f,     0.1f), // EXTRALARGE
    SchoolFlockingParams(1.5f,  16.0f, 6.0f, ms_tankMin, ms_tankMax,  3.50f,  0.01f,   0.8f,    0.1f,   0.1f,    0.10f,      0.5f,     1.0f ), // LARGESLOW
    SchoolFlockingParams(2.5f,  16.0f, 6.0f, ms_tankMin, ms_tankMax,  3.00f,  0.01f,   0.7f,    0.1f,   0.1f,    0.15f,      0.5f,     1.0f ), // LARGE
    SchoolFlockingParams(3.5f,  12.0f, 5.0f, ms_tankMin, ms_tankMax,  2.50f,  0.01f,   0.6f,    0.2f,   0.2f,    0.10f,      0.5f,     1.0f ), // LARGEFAST
    SchoolFlockingParams(2.5f,  14.0f, 5.0f, ms_tankMin, ms_tankMax,  2.00f,  0.01f,   0.5f,    0.1f,   0.1f,    0.15f,      0.5f,     2.0f ), // MEDIUMSLOW
    SchoolFlockingParams(3.5f,  12.0f, 4.0f, ms_tankMin, ms_tankMax,  1.60f,  0.01f,   0.4f,    0.1f,   0.1f,    0.15f,      0.5f,     2.0f ), // MEDIUM
    SchoolFlockingParams(6.0f,  10.0f, 3.0f, ms_tankMin, ms_tankMax,  1.40f,  0.01f,   0.3f,    0.2f,   0.1f,    0.10f,      0.5f,     2.0f ), // MEDIUMFAST
    SchoolFlockingParams(5.0f,  10.0f,10.0f, ms_tankMin, ms_tankMax,  1.50f,  0.01f,   0.1f,    0.1f,   0.1f,    0.15f,      0.5f,     3.0f ), // MEDIUMSPARSE
    SchoolFlockingParams(3.0f,   8.0f, 3.0f, ms_tankMin, ms_tankMax,  1.00f,  0.01f,   0.2f,    0.1f,   0.2f,    0.10f,      0.5f,     4.0f ), // SMALLSLOW
    SchoolFlockingParams(5.0f,   5.0f, 2.0f, ms_tankMin, ms_tankMax,  0.25f,  0.01f,   0.1f,    0.1f,   0.4f,    0.15f,      0.5f,     5.0f ), // SMALL
    SchoolFlockingParams(7.0f,   4.0f, 1.0f, ms_tankMin, ms_tankMax,  0.25f,  0.01f,   0.1f,    0.2f,   0.5f,    0.40f,      0.1f,     6.0f )  // SMALLFAST
};

ThreadedRenderingVk::SchoolDesc ThreadedRenderingVk::ms_schoolInfo[MODEL_COUNT] = 
{
    // ModelId,           NumFish, Scale, 
    { MODEL_BLACK_WHITE_FISH,  75, 2.00f, ms_fishTypeDefs[FISHTYPE_LARGEFAST] },
    { MODEL_BLUE_FISH,         75, 2.00f, ms_fishTypeDefs[FISHTYPE_LARGE] },          
    { MODEL_BLUE_FISH_02,     100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUM] },
    { MODEL_BLUE_FISH_03,     100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUMSLOW] },
    { MODEL_BLUE_FISH_04,     100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUM] },
    { MODEL_BLUE_FISH_05,     100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUM] },
    { MODEL_BLUE_FISH_06,     100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUMFAST] },
    { MODEL_BLUE_FISH_07,     200, 0.50f, ms_fishTypeDefs[FISHTYPE_SMALLFAST] },
    { MODEL_BLUE_FISH_08,     200, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUMSPARSE] },
    { MODEL_BLUE_FISH_09,      50, 3.00f, ms_fishTypeDefs[FISHTYPE_EXTRALARGE] },
    { MODEL_CYAN_FISH,        100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUM] },
    { MODEL_PINK_FISH,        150, 0.75f, ms_fishTypeDefs[FISHTYPE_SMALLSLOW] },
    { MODEL_RED_FISH,         50,  3.00f, ms_fishTypeDefs[FISHTYPE_LARGESLOW] },
    { MODEL_VIOLET_FISH,      250, 0.50f, ms_fishTypeDefs[FISHTYPE_SMALLFAST] },
    { MODEL_YELLOW_FISH,      100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUMFAST] },
    { MODEL_YELLOW_FISH_02,   100, 1.50f, ms_fishTypeDefs[FISHTYPE_MEDIUMSLOW] },
    { MODEL_YELLOW_FISH_03,   100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUMFAST] },
    { MODEL_YELLOW_FISH_04,   100, 0.75f, ms_fishTypeDefs[FISHTYPE_MEDIUMFAST] },
    { MODEL_YELLOW_FISH_05,   150, 0.80f, ms_fishTypeDefs[FISHTYPE_SMALLSLOW] },
    { MODEL_YELLOW_FISH_06,   100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUM] },
    { MODEL_YELLOW_FISH_07,   100, 1.20f, ms_fishTypeDefs[FISHTYPE_MEDIUMSLOW] },
    { MODEL_YELLOW_FISH_08,   100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUM] },
    { MODEL_YELLOW_FISH_09,   150, 0.80f, ms_fishTypeDefs[FISHTYPE_SMALLSLOW] },
    { MODEL_YELLOW_FISH_10,   150, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUMSPARSE] },
    { MODEL_YELLOW_FISH_11,   100, 1.00f, ms_fishTypeDefs[FISHTYPE_MEDIUM] }
};

//-----------------------------------------------------------------------------
// FUNCTION NEEDED BY THE FRAMEWORK

NvAppBase* NvAppFactory()
{
    return new ThreadedRenderingVk();
}
