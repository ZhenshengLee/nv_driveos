//----------------------------------------------------------------------------------
// File:        NvVkUtil/NvModelExtVK.cpp
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

#include "NvModelExtVK.h"
#include "NvModelExt.h"
#include "NvModelExtObj.h"

namespace Nv
{

	NvModelExtVK* NvModelExtVK::Create(NvVkContext& vk, NvModelExt* pSourceModel)
	{
		if (NULL == pSourceModel)
		{
			return NULL;
		}

		NvModelExtVK* model = new NvModelExtVK(pSourceModel);
		model->m_pSourceModel = pSourceModel;
		model->PrepareForRendering(vk, pSourceModel);
		return model;
	}

	void NvModelExtVK::Release(NvVkContext& vk)
	{
	}

	NvModelExtVK::NvModelExtVK(NvModelExt* pSourceModel) :
		m_pSourceModel(pSourceModel),
		m_instanced(false)
	{
		NV_ASSERT(NULL != pSourceModel);
	}

	NvModelExtVK::~NvModelExtVK()
	{
		if (m_pSourceModel)
		{
			delete m_pSourceModel;
		}
	};

	bool NvModelExtVK::UpdateBoneTransforms()
	{
		std::vector<NvMeshExtVK*>::iterator meshIt = m_meshes.begin();
		std::vector<NvMeshExtVK*>::iterator meshEnd = m_meshes.end();
		bool result = true;

		for (; meshIt != meshEnd; ++meshIt)
		{
			result &= (*meshIt)->UpdateBoneTransforms(GetModel()->GetSkeleton());
		}
		return result;
	}

#define EMBEDDED_RESOURCES 1

#if defined(EMBEDDED_RESOURCES)

#include "assets/textures/Black_White_Fish_dds.h"
#include "assets/textures/Blue_Fish_dds_04.h" 
#include "assets/textures/Blue_Fish_dds_08.h"
#include "assets/textures/caustic1_dds.h" 
#include "assets/textures/Pink_Fish_dds.h"
#include "assets/textures/Yellow_Fish_02_dds.h" 
#include "assets/textures/Yellow_Fish_06_dds.h" 
#include "assets/textures/Yellow_Fish_10_dds.h"
#include "assets/textures/Blue_Fish_dds_01.h"     
#include "assets/textures/Blue_Fish_dds_05.h" 
#include "assets/textures/Blue_Fish_dds_09.h"
#include "assets/textures/caustic2_dds.h"  
#include "assets/textures/Red_Fish_dds.h"
#include "assets/textures/Yellow_Fish_03_dds.h" 
#include "assets/textures/Yellow_Fish_07_dds.h" 
#include "assets/textures/Yellow_Fish_11_dds.h"
#include "assets/textures/Blue_Fish_dds_02.h"     
#include "assets/textures/Blue_Fish_dds_06.h" 
#include "assets/textures/Blue_Fish_dds_10.h"
#include "assets/textures/Cyan_Fish_dds.h"
#include "assets/textures/sand_dds.h"
#include "assets/textures/Yellow_Fish_04_dds.h" 
#include "assets/textures/Yellow_Fish_08_dds.h" 
#include "assets/textures/Yellow_Fish_12_dds.h"
#include "assets/textures/Blue_Fish_dds_03.h"     
#include "assets/textures/Blue_Fish_dds_07.h"
#include "assets/textures/Blue_Fish_dds.h"
#include "assets/textures/gradient_dds.h"
#include "assets/textures/Violet_Fish_dds.h"
#include "assets/textures/Yellow_Fish_05_dds.h"
#include "assets/textures/Yellow_Fish_09_dds.h"
#include "assets/textures/Yellow_Fish_dds.h"

	struct textureDataLookupTable
	{
		const char* m_filename;
		unsigned char *m_data;
		unsigned int m_length;
	};

	static struct textureDataLookupTable textureTable[]
	{
		{"textures/Black_White_Fish.dds", Black_White_Fish_dds, Black_White_Fish_dds_len},
		{"textures/Blue_Fish_05.dds", Blue_Fish_05_dds, Blue_Fish_05_dds_len},
		{"textures/Blue_Fish_09.dds", Blue_Fish_09_dds, Blue_Fish_09_dds_len},
		{"textures/Pink_Fish.dds", Pink_Fish_dds, Pink_Fish_dds_len},
		{"textures/Violet_Fish.dds", Violet_Fish_dds, Violet_Fish_dds_len},
		{"textures/Yellow_Fish_05.dds", Yellow_Fish_05_dds, Yellow_Fish_05_dds_len},
		{"textures/Yellow_Fish_09.dds", Yellow_Fish_09_dds, Yellow_Fish_09_dds_len},
		{"textures/Blue_Fish_02.dds", Blue_Fish_02_dds, Blue_Fish_02_dds_len},
		{"textures/Blue_Fish_06.dds", Blue_Fish_06_dds, Blue_Fish_06_dds_len},
		{"textures/Blue_Fish.dds", Blue_Fish_dds, Blue_Fish_dds_len},
		{"textures/Red_Fish.dds", Red_Fish_dds, Red_Fish_dds_len},
		{"textures/Yellow_Fish_02.dds", Yellow_Fish_02_dds, Yellow_Fish_02_dds_len},
		//there is a typo in the original filename, extra space
		{"textures/Yellow_Fish_06.dds", Yellow_Fish_06_dds, Yellow_Fish_06_dds_len},
		{"textures/Yellow_Fish_10.dds", Yellow_Fish_10_dds, Yellow_Fish_10_dds_len},
		{"textures/Blue_Fish_03.dds", Blue_Fish_03_dds, Blue_Fish_03_dds_len},
		{"textures/Blue_Fish_07.dds", Blue_Fish_07_dds, Blue_Fish_07_dds_len},
		{"textures/Cyan_Fish.dds", Cyan_Fish_dds, Cyan_Fish_dds_len},
		{"textures/Yellow_Fish_03.dds", Yellow_Fish_03_dds, Yellow_Fish_03_dds_len},
		{"textures/Yellow_Fish_07.dds", Yellow_Fish_07_dds, Yellow_Fish_07_dds_len},
		{"textures/Yellow_Fish_11.dds", Yellow_Fish_11_dds, Yellow_Fish_11_dds_len},
        {"textures/Blue_Fish_04.dds", Blue_Fish_04_dds, Blue_Fish_04_dds_len},
        {"textures/Blue_Fish_08.dds", Blue_Fish_08_dds, Blue_Fish_08_dds_len},
        {"textures/Yellow_Fish_04.dds", Yellow_Fish_04_dds, Yellow_Fish_04_dds_len},
        {"textures/Yellow_Fish_08.dds", Yellow_Fish_08_dds, Yellow_Fish_08_dds_len},
        {"textures/Yellow_Fish.dds", Yellow_Fish_dds, Yellow_Fish_08_dds_len}
	};

	unsigned char * findDataByFilename(const char *filename, unsigned int &length);
	unsigned char * findDataByFilename(const char *filename, unsigned int &length)
	{
		for(int i = 0; i < 25; ++i)
		{
			if (strcmp(textureTable[i].m_filename, filename) == 0)
			{
				length = textureTable[i].m_length;
				return textureTable[i].m_data;
			}
		}

		length = 0;
		return NULL;
	}
#endif

	void NvModelExtVK::PrepareForRendering(NvVkContext& vk,
		NvModelExt* pModel)
	{
		

		VkSamplerCreateInfo samplerCreateInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
		samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
		samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
		samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCreateInfo.mipLodBias = 0.0;
		samplerCreateInfo.maxAnisotropy = 1;
		samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
		samplerCreateInfo.minLod = 0.0;
		samplerCreateInfo.maxLod = 16.0;
		samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

		vkCreateSampler(vk.device(), &samplerCreateInfo, 0, &m_linearSampler);

		// Get GL usable versions of all the textures used by the model
		uint32_t textureCount = m_pSourceModel->GetTextureCount();
		m_textures.resize(textureCount);
		for (uint32_t textureIndex = 0; textureIndex < textureCount; ++textureIndex)
		{
			NvVkTexture* t = new NvVkTexture;
			
			#if defined(EMBEDDED_RESOURCES)

			unsigned int length = 0;
			unsigned char *dataBuf = findDataByFilename(m_pSourceModel->GetTextureName(textureIndex).c_str(), length);

			if (dataBuf)
			{
				if (vk.uploadTextureFromDDSData((const char*)dataBuf, length, *t))
				{
					m_textures[textureIndex] = t;
				}
				else
				{
					delete t;
				}
			}
			else
			{
				delete t;
			}

			#else
			if (vk.uploadTextureFromFile(m_pSourceModel->GetTextureName(textureIndex).c_str(), *t)) 
			{
				m_textures[textureIndex] = t;
			}
			else
			{
				delete t;
			}
			#endif
		}

		// Get VK usable versions of all the materials in the model
		uint32_t materialCount = pModel->GetMaterialCount();
		m_materials.resize(materialCount);
		if (materialCount > 0)
		{
			for (uint32_t materialIndex = 0; materialIndex < materialCount; ++materialIndex)
			{
				m_materials[materialIndex].InitFromMaterial(m_pSourceModel, materialIndex);
			}
		}

		// Get VK renderable versions of all meshes in the model
		uint32_t meshCount = pModel->GetMeshCount();
		m_meshes.resize(meshCount);
		if (meshCount > 0)
		{
			for (uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex)
			{
				m_meshes[meshIndex] = new NvMeshExtVK;
				m_meshes[meshIndex]->InitFromSubmesh(vk, pModel, meshIndex);
			}
		}

		InitVertexState();
	}

	bool NvModelExtVK::EnableInstanceData(uint32_t instanceVertSize) {
		uint32_t meshCount = GetMeshCount();
		for (uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex)
		{
			if (!GetMesh(meshIndex)->EnableInstanceData(instanceVertSize))
				return false;
		}

		m_instanced = true;
		return true;
	}

	bool NvModelExtVK::AddInstanceData(uint32_t location, VkFormat format, uint32_t offset) {
		uint32_t meshCount = GetMeshCount();
		for (uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex)
		{
			if (!GetMesh(meshIndex)->AddInstanceData(location, format, offset))
				return false;
		}

		return true;
	}

	bool NvModelExtVK::InitVertexState() {
		uint32_t meshCount = GetMeshCount();
		for (uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex)
		{
			if (!GetMesh(meshIndex)->InitVertexState())
				return false;
		}

		return true;
	}
}
