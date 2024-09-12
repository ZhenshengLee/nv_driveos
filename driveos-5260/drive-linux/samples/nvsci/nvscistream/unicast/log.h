//
// Logging utility
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef LOG_H
#define LOG_H

#include <iostream>
#include <cstdlib>

#define LOG_MSG(s) std::cout << s << std::endl;

#define LOG_ERR(s)                                                              \
    {                                                                           \
        std::cout << __FILE__ << " (" << std::dec << __LINE__ << "), "          \
        << __func__ << ": " << s << std::endl;                                  \
    }

#define DEBUG_ENABLE 1
#if DEBUG_ENABLE
#define LOG_DEBUG(s) LOG_MSG(s);
#else
#define LOG_DEBUG(s)
#endif

#define DO_EXIT() exit(EXIT_FAILURE)

#define LOG_ERR_EXIT(s)     \
    {                       \
        LOG_ERR(s);         \
        DO_EXIT();          \
    }

#define CHECK_NVSCIERR(e)                                       \
    {                                                           \
        auto ret = (e);                                         \
        if (ret != NvSciError_Success)                          \
        {                                                       \
            LOG_ERR("NvSci error " << std::hex << ret);       \
            DO_EXIT();                                          \
        }                                                       \
    }

#define CHECK_NVMEDIAERR(e)                                     \
    {                                                           \
        auto ret = (e);                                         \
        if (ret != NVMEDIA_STATUS_OK)                           \
        {                                                       \
            LOG_ERR("NvMedia error " << std::hex << ret);     \
            DO_EXIT();                                          \
        }                                                       \
    }

#define CHECK_CUDAERR(e)                                        \
    {                                                           \
        auto ret = (e);                                         \
        if (ret != CUDA_SUCCESS)                                \
        {                                                       \
            LOG_ERR("CUDA error " << std::hex << ret);        \
            DO_EXIT();                                          \
        }                                                       \
    }


#endif
