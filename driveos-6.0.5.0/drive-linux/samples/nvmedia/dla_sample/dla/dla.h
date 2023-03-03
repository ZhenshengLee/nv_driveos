/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _DLA_H_
#define _DLA_H_

#include <memory>
#include <vector>
#include <array>

#include "nvscisync.h"
#include "nvmedia_dla.h"
#include "nvmedia_dla_nvscisync.h"
#include "nvmedia_core.h"

#include "tensor.h"

#include <limits>

//! Dla class
//! Dla class abstract NvMediaDla APIs and provide functions to load loadable and
//! execute loadable with provided input data.

class Dla final
{
public:
    static const uint32_t INVALID_LOADABLEINDEX = std::numeric_limits<uint32_t>::max();

    static NvMediaStatus GetDlaVersion(NvMediaVersion *version);

    static NvMediaStatus PingById(const uint32_t dlaId);

    static std::unique_ptr<Dla> Create();

    ~Dla();

    NvMediaStatus Init(uint32_t dlaId, uint32_t numTasks);

    //! One Dla class can hold only one loadable.
    NvMediaStatus AddLoadable(std::string profileName, uint32_t &loadableIndex);

    NvMediaStatus GetDesc(
        uint32_t loadableIndex,
        std::vector<NvMediaDlaTensorDescriptor> &vInputTensorDesc,
        std::vector<NvMediaDlaTensorDescriptor> &vOutputTensorDesc
    );

    NvMediaStatus DataRegister(
        uint32_t loadableIndex,
        Tensor *tensor
    );

    NvMediaStatus DataUnregister(
        uint32_t loadableIndex,
        Tensor *tensor
    );

    NvMediaStatus RemoveLoadable(uint32_t loadableIndex);

    NvMediaStatus Submit(
        uint32_t loadableIndex,
        std::vector<Tensor*>  &vpInputTensor,
        std::vector<Tensor*>  &vpOutputTensor
    );

    //SciSync related api
    NvMediaStatus GetAttrList(
                    NvSciSyncModule module,
                    NvSciSyncAttrList &attrList,
                    NvMediaNvSciSyncClientType syncType);

    NvMediaStatus RegisterSyncObj(
                    NvMediaNvSciSyncObjType syncObjType,
                    NvSciSyncObj syncObj);

    NvMediaStatus UnRegisterSyncObj(
                    NvSciSyncObj syncObj);

    NvMediaStatus SetEOFSyncObj(
                    NvSciSyncObj syncObj);

    NvMediaStatus InsertPreSciFences(NvSciSyncFence *EOFfence);

    NvMediaStatus GetEOFSciFences(
                    NvSciSyncObj eofSyncObj,
                    NvSciSyncFence *EOFfence);

protected:
    NvMediaStatus PrintTensorDesc(NvMediaDlaTensorDescriptor *tensorDesc);

private:

    Dla(NvMediaDla *m_pDla);

    NvMediaDla *m_pDla;

    std::vector<NvMediaDlaLoadable*> m_vLoadables;

    static const std::size_t MAX_NUM_OF_DLA_DATA = 40;

    std::array<NvMediaDlaData, MAX_NUM_OF_DLA_DATA> m_aInputDlaData;

    std::array<NvMediaDlaData, MAX_NUM_OF_DLA_DATA> m_aOutputDlaData;
};

#endif // END OF _DLA_H_
