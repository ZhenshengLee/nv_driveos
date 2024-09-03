/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CUTILS_HPP
#define CUTILS_HPP

// Standard header files
#include <iostream>
#include <cstdarg>
#include <memory>
#include <vector>

// SIPL header files
#include "NvSIPLCommon.hpp"
#include "NvSIPLPipelineMgr.hpp"

// Other NVIDIA header files
#include "nvmedia_core.h"
#include "nvscierror.h"

using namespace nvsipl;

struct EventMap {
    NvSIPLPipelineNotifier::NotificationType eventType;
    const char *eventName;
};

const static uint32_t MAX_NUM_SENSORS = 16U;
const static uint32_t NUM_SYNC_INTERFACES = 2U;
const static uint32_t NUM_SYNC_ACTORS = 3U;

const static uint32_t RECON_NUM = 3U;
const static uint32_t BUFFER_NUM = 2U;

struct CloseNvSciBufAttrList {
    void operator()(NvSciBufAttrList *attrList) const {
        if (attrList != nullptr) {
            if ((*attrList) != nullptr) {
                NvSciBufAttrListFree(*attrList);
            }
            delete attrList;
        }
    }
};

struct CloseNvSciSyncAttrList {
    void operator()(NvSciSyncAttrList *attrList) const {
        if (attrList != nullptr) {
            if ((*attrList) != nullptr) {
                NvSciSyncAttrListFree(*attrList);
            }
            delete attrList;
        }
    }
};

struct CloseNvSciSyncObj {
    void operator()(NvSciSyncObj *syncObj) const {
        if (syncObj != nullptr) {
            if ((*syncObj) != nullptr) {
                NvSciSyncObjFree(*syncObj);
            }
            delete syncObj;
        }
    }
};

// Helper macros
#define ARRAY_SIZE(a) \
    (sizeof(a)/sizeof((a)[0]))

#define CHK_PTR_AND_RETURN(ptr, api) \
    if ((ptr) == nullptr) { \
        LOG_ERR("%s failed\n", (api)); \
        return NVSIPL_STATUS_OUT_OF_MEMORY; \
    }

#define CHK_PTR_AND_RETURN_BADARG(ptr, name) \
    if ((ptr) == nullptr) { \
        LOG_ERR("%s is null\n", (name)); \
        return NVSIPL_STATUS_BAD_ARGUMENT; \
    }

#define CHK_STATUS_AND_RETURN(status, api) \
    if ((status) != NVSIPL_STATUS_OK) { \
        LOG_ERR("%s failed, status: %u\n", (api), (status)); \
        return (status); \
    }

#define CHK_NVMSTATUS_AND_RETURN(nvmStatus, api) \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) { \
        LOG_ERR("%s failed, status: %u\n", (api), (nvmStatus)); \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_NVSCISTATUS_AND_RETURN(nvSciStatus, api) \
    if (nvSciStatus != NvSciError_Success) { \
        LOG_ERR("%s failed, status: %u\n", (api), (nvSciStatus)); \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_WFDSTATUS_AND_RETURN(wfdStatus, api) \
    if (wfdStatus) { \
        LOG_ERR("%s failed, status: %u\n", (api), (wfdStatus)); \
        return NVSIPL_STATUS_ERROR; \
    }

#define GET_WFDERROR_AND_RETURN(device) \
    { \
        WFDErrorCode wfdErr = wfdGetError(device); \
        if (wfdErr) { \
            LOG_ERR("WFD error %x, line: %u\n", wfdErr, __LINE__); \
            return NVSIPL_STATUS_ERROR; \
        } \
    }

#define CHK_STATUS_AND_EXIT(status, api) \
    if ((status) != NVSIPL_STATUS_OK) { \
        LOG_ERR("%s failed, status: %u\n", (api), (status)); \
        return; \
    }

#define CHK_NVMSTATUS_AND_EXIT(nvmStatus, api) \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) { \
        LOG_ERR("%s failed, status: %u\n", (api), (nvmStatus)); \
        return; \
    }

#define LINE_INFO __FUNCTION__, __LINE__

#define LOG_LEVEL(...) \
    CLogger::GetInstance().SetLogLevel(__VA_ARGS__)

//! Log a message at debugging level.
#define LOG_DBG(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_DBG, LINE_INFO, __VA_ARGS__)

//! Log a message at info level.
#define LOG_INFO(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_INFO, LINE_INFO, __VA_ARGS__)

//! Log a message at warning level.
#define LOG_WARN(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_WARN, LINE_INFO, __VA_ARGS__)

//! Log a message at error level.
#define LOG_ERR(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_ERR, LINE_INFO, __VA_ARGS__)

//! Log a message at preset level.
#define LOG_MSG(...) \
    CLogger::GetInstance().LogMessage(__VA_ARGS__)

#define LEVEL_NONE CLogger::LogLevel::LEVEL_NO_LOG

#define LEVEL_ERR CLogger::LogLevel::LEVEL_ERROR

#define LEVEL_WARN CLogger::LogLevel::LEVEL_WARNING

#define LEVEL_INFO CLogger::LogLevel::LEVEL_INFORMATION

#define LEVEL_DBG CLogger::LogLevel::LEVEL_DEBUG

//! \brief Logger utility class
//! This is a singleton class - at most one instance can exist at any time.
class CLogger
{
public:
    enum LogLevel {
        LEVEL_NO_LOG = 0U,
        LEVEL_ERROR,
        LEVEL_WARNING,
        LEVEL_INFORMATION,
        LEVEL_DEBUG
    };

    enum LogStyle {
        LOG_STYLE_NORMAL = 0U,
        LOG_STYLE_FUNCTION_LINE = 1U
    };

    //! Get the logging instance.
    //! \return Reference to the logger object.
    static CLogger& GetInstance();

    //! Set the level for logging.
    //! \param[in] eLevel The logging level.
    void SetLogLevel(LogLevel eLevel);

    //! Get the level for logging.
    //! \return The logging level.
    LogLevel GetLogLevel();

    //! Set the style for logging.
    //! \param[in] eStyle The logging style.
    void SetLogStyle(LogStyle eStyle);

    //! Log a message (C string).
    //! \param[in] eLevel The logging level.
    //! \param[in] pszFunctionName Name of the function as a C string.
    //! \param[in] sLineNumber Line number.
    //! \param[in] pszFormat Format string as a C string.
    void LogLevelMessage(LogLevel eLevel,
                         const char *pszFunctionName,
                         uint32_t sLineNumber,
                         const char *pszFormat,
                         ...);

    //! Log a message (C++ string).
    //! \param[in] eLevel The logging level.
    //! \param[in] sFunctionName Name of the function as a C++ string.
    //! \param[in] sLineNumber Line number.
    //! \param[in] sFormat Format string as a C++ string.
    void LogLevelMessage(LogLevel eLevel,
                         std::string sFunctionName,
                         uint32_t sLineNumber,
                         std::string sFormat,
                         ...);

    //! Log a message (C string) at preset level.
    //! \param[in] pszFormat Format string as a C string.
    void LogMessage(const char *pszFormat, ...);

    //! Log a message (C++ string) at preset level.
    //! \param[in] sFormat Format string as a C++ string.
    void LogMessage(std::string sFormat, ...);

private:
    // Need private constructor because this is a singleton
    CLogger() = default;
    LogLevel m_level = LEVEL_ERR;
    LogStyle m_style = LOG_STYLE_NORMAL;

    void LogLevelMessageVa(LogLevel eLevel,
                           const char *pszFunctionName,
                           uint32_t sLineNumber,
                           const char *pszFormat,
                           va_list ap);
    void LogMessageVa(const char *pszFormat, va_list ap);
};

//! \brief Helper class for managing files
class CFileManager
{
public:
    CFileManager(std::string const &name, std::string const &mode) : m_name(name), m_mode(mode)
    {
    }

    CFileManager() = delete;

    FILE * GetFile()
    {
        if (m_file == nullptr) {
            m_file = fopen(m_name.c_str(), m_mode.c_str());
        }
        return m_file;
    }

    std::string const & GetName()
    {
        return m_name;
    }

    ~CFileManager()
    {
        if (m_file != nullptr) {
            fclose(m_file);
        }
    }

private:
    FILE *m_file = nullptr;
    const std::string m_name;
    const std::string m_mode;
};

//! \brief Loads NITO file for given camera module
//! The function assumes that the NITO file has the same name as the camera module.
//! If the module-specific NITO file is not found, default.nito is loaded instead.
SIPLStatus LoadNitoFile(std::string const &folderPath,
                        std::string const &moduleName,
                        std::vector<uint8_t> &nito,
                        bool &defaultLoaded);

//! \brief Provides a string that names the event type
SIPLStatus GetEventName(const NvSIPLPipelineNotifier::NotificationData &event, const char *&eventName);

#endif // CUTILS_HPP
