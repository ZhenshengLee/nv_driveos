/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Standard header files
#include <cstring>

// Sample application header files
#include "CUtils.hpp"

CLogger& CLogger::GetInstance()
{
    static CLogger instance;
    return instance;
}

void CLogger::SetLogLevel(LogLevel level)
{
    m_level = (level > LEVEL_DBG) ? LEVEL_DBG : level;
}

CLogger::LogLevel CLogger::GetLogLevel()
{
    return m_level;
}

void CLogger::SetLogStyle(LogStyle style)
{
    m_style = (style > LOG_STYLE_FUNCTION_LINE) ? LOG_STYLE_FUNCTION_LINE : style;
}

void CLogger::LogLevelMessageVa(LogLevel level,
                                const char *functionName,
                                uint32_t lineNumber,
                                const char *format,
                                va_list ap)
{
    char str[256] = {'\0',};

    if (level > m_level) {
        return;
    }

    strcpy(str, "nvsipl_sample: ");
    switch (level) {
        case LEVEL_NONE:
            break;
        case LEVEL_ERR:
            strcat(str, "ERROR: ");
            break;
        case LEVEL_WARN:
            strcat(str, "WARNING: ");
            break;
        case LEVEL_INFO:
            break;
        case LEVEL_DBG:
            break;
    }

    vsnprintf(str + strlen(str), sizeof(str) - strlen(str), format, ap);

    if (m_style == LOG_STYLE_NORMAL) {
        if ((strlen(str) != 0) && (str[strlen(str) - 1] == '\n')) {
            strcat(str, "\n");
        }
    } else if (m_style == LOG_STYLE_FUNCTION_LINE) {
        if ((strlen(str) != 0) && (str[strlen(str) - 1] == '\n')) {
            str[strlen(str) - 1] = '\0';
        }
        snprintf(str + strlen(str),
                 sizeof(str) - strlen(str),
                 " at %s():%d\n",
                 functionName,
                 lineNumber);
    }

    std::cout << str;
}

void CLogger::LogLevelMessage(LogLevel level,
                              const char *functionName,
                              uint32_t lineNumber,
                              const char *format,
                              ...)
{
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName, lineNumber, format, ap);
    va_end(ap);
}

void CLogger::LogLevelMessage(LogLevel level,
                              std::string functionName,
                              uint32_t lineNumber,
                              std::string format,
                              ...)
{
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName.c_str(), lineNumber, format.c_str(), ap);
    va_end(ap);
}

void CLogger::LogMessageVa(const char *format, va_list ap)
{
    char str[128] = {'\0',};
    vsnprintf(str, sizeof(str), format, ap);
    std::cout << str;
}

void CLogger::LogMessage(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogMessageVa(format, ap);
    va_end(ap);
}

void CLogger::LogMessage(std::string format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogMessageVa(format.c_str(), ap);
    va_end(ap);
}

SIPLStatus LoadNitoFile(std::string const &folderPath,
                        std::string const &moduleName,
                        std::vector<uint8_t> &nito,
                        bool &defaultLoaded)
{
    CFileManager moduleNito(folderPath + moduleName + ".nito", "rb");
    CFileManager defaultNito(folderPath + "default.nito", "rb");
    FILE *fp = nullptr;
    defaultLoaded = false;

    if (moduleNito.GetFile() != nullptr) {
        LOG_MSG("Opened NITO file for module \"%s\"\n", moduleName.c_str());
        fp = moduleNito.GetFile();
    } else {
        LOG_INFO("File \"%s\" not found\n", moduleNito.GetName().c_str());
        LOG_ERR("Unable to open NITO file for module \"%s\", image quality is not supported\n",
                moduleName.c_str());
        if (defaultNito.GetFile() != nullptr) {
            LOG_MSG("Opened default NITO file for module \"%s\"\n", defaultNito.GetName().c_str());
            fp = defaultNito.GetFile();
            defaultLoaded = true;
        } else {
            LOG_ERR("Unable to open default NITO file \"%s\"\n", defaultNito.GetName().c_str());
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
    }

    // Check file size
    fseek(fp, 0, SEEK_END);
    size_t fsize = ftell(fp);
    rewind(fp);

    if (fsize <= 0U) {
        LOG_ERR("NITO file for module \"%s\" is of invalid size\n", moduleName.c_str());
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    // Allocate blob memory
    nito.resize(fsize);

    // Load NITO
    size_t result = fread(nito.data(), 1U, fsize, fp);
    if (result != fsize) {
        LOG_ERR("Unable to read data from NITO file for module \"%s\"" \
                ", image quality is not supported\n",
                moduleName.c_str());
        nito.resize(0U);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    LOG_INFO("Data from NITO file loaded for module \"%s\"\n", moduleName.c_str());

    return NVSIPL_STATUS_OK;
}

SIPLStatus GetEventName(const NvSIPLPipelineNotifier::NotificationData &event, const char *&eventName)
{
    static const EventMap eventNameTable[] = {
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_INFO_ICP_PROCESSING_DONE,
            "NOTIF_INFO_ICP_PROCESSING_DONE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_INFO_ISP_PROCESSING_DONE,
            "NOTIF_INFO_ISP_PROCESSING_DONE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_INFO_ACP_PROCESSING_DONE,
            "NOTIF_INFO_ACP_PROCESSING_DONE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_INFO_ICP_AUTH_SUCCESS,
            "NOTIF_INFO_ICP_AUTH_SUCCESS"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_INFO_CDI_PROCESSING_DONE,
            "NOTIF_INFO_CDI_PROCESSING_DONE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_FRAME_DROP,
            "NOTIF_WARN_ICP_FRAME_DROP"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_FRAME_DISCONTINUITY,
            "NOTIF_WARN_ICP_FRAME_DISCONTINUITY"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_CAPTURE_TIMEOUT,
            "NOTIF_WARN_ICP_CAPTURE_TIMEOUT"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ICP_BAD_INPUT_STREAM,
            "NOTIF_ERROR_ICP_BAD_INPUT_STREAM"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ICP_CAPTURE_FAILURE,
            "NOTIF_ERROR_ICP_CAPTURE_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE,
            "NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ISP_PROCESSING_FAILURE,
            "NOTIF_ERROR_ISP_PROCESSING_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ACP_PROCESSING_FAILURE,
            "NOTIF_ERROR_ACP_PROCESSING_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE,
            "NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_INTERNAL_FAILURE,
            "NOTIF_ERROR_INTERNAL_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ICP_AUTH_FAILURE,
            "NOTIF_ERROR_ICP_AUTH_FAILURE"}
    };

    for (uint32_t i = 0U; i < ARRAY_SIZE(eventNameTable); i++) {
        if (event.eNotifType == eventNameTable[i].eventType) {
            eventName = eventNameTable[i].eventName;
            return NVSIPL_STATUS_OK;
        }
    }

    LOG_ERR("Unknown event type\n");
    return NVSIPL_STATUS_BAD_ARGUMENT;
}
