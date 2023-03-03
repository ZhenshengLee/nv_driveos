/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CUtils.hpp"
#include <cstring>
#include <iomanip>


#ifndef NITO_PATH
    #ifdef NVMEDIA_QNX
        #define NITO_PATH "/proc/boot/"
    #else
        #define NITO_PATH "/opt/nvidia/nvmedia/nit/"
    #endif
#endif

using namespace std;

// Log utils
CLogger& CLogger::GetInstance()
{
    static CLogger instance;
    return instance;
}

void CLogger::SetLogLevel(LogLevel level)
{
    m_level = (level > LEVEL_DBG) ? LEVEL_DBG : level;
}

CLogger::LogLevel CLogger::GetLogLevel(void)
{
    return m_level;
}

void CLogger::SetLogStyle(LogStyle style)
{
    m_style = (style > LOG_STYLE_FUNCTION_LINE) ? LOG_STYLE_FUNCTION_LINE
                                                : style;
}

void CLogger::LogLevelMessageVa(LogLevel level, const char *functionName,
                                       uint32_t lineNumber, const char *format,
                                                                    va_list ap)
{
    char str[256] = {'\0',};

    if (level > m_level) {
        return;
    }

    strcpy(str, "nvsipl_camera: ");
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
            // Empty
            break;
    }

    vsnprintf(str + strlen(str), sizeof(str) - strlen(str), format, ap);

    if (m_style == LOG_STYLE_NORMAL) {
        if (strlen(str) != 0 && str[strlen(str) - 1] == '\n') {
            strcat(str, "\n");
        }
    } else if (m_style == LOG_STYLE_FUNCTION_LINE) {
        if (strlen(str) != 0 && str[strlen(str) - 1] == '\n') {
            str[strlen(str) - 1] = 0;
        }
        snprintf(str + strlen(str), sizeof(str) - strlen(str), " at %s():%d\n",
                                                     functionName, lineNumber);
    }

    cout << str;
}

void CLogger::LogLevelMessage(LogLevel level, const char *functionName,
                               uint32_t lineNumber, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName, lineNumber, format, ap);
    va_end(ap);
}

void CLogger::LogLevelMessage(LogLevel level, std::string functionName,
                               uint32_t lineNumber, std::string format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName.c_str(), lineNumber,
                                                       format.c_str(), ap);
    va_end(ap);
}

void CLogger::LogMessageVa(const char *format, va_list ap)
{
    char str[128] = {'\0',};
    vsnprintf(str, sizeof(str), format, ap);
    cout << str;
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

SIPLStatus CUtils::IsRawBuffer(NvSciBufObj bufObj, bool &bIsRaw)
{
    bIsRaw = false;
    BufferAttrs bufAttrs;
    SIPLStatus status = PopulateBufAttr(bufObj, bufAttrs);
    CHK_STATUS_AND_RETURN(status, "PopulateBufAttr");
    NvSciBufAttrValColorFmt colorFmt = bufAttrs.planeColorFormats[0];
    if ((colorFmt >= NvSciColor_Bayer8RGGB) && (colorFmt <= NvSciColor_Signed_X12Bayer20GBRG)) {
        bIsRaw = true;
    }
    return NVSIPL_STATUS_OK;
}

SIPLStatus CUtils::CreateRgbaBuffer(NvSciBufModule &bufModule,
                                    NvSciBufAttrList &bufAttrList,
                                    uint32_t width,
                                    uint32_t height,
                                    NvSciBufObj *pBufObj)
{
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrValImageScanType bufScanType = NvSciBufScan_ProgressiveType;
    bool imgCpuAccess = true;
    bool imgCpuCacheEnabled = true;
    uint32_t planeCount = 1U;
    NvSciBufAttrValColorFmt planeColorFmt = NvSciColor_A8B8G8R8;
    NvSciBufAttrValColorStd planeColorStd = NvSciColorStd_SRGB;
    NvSciBufAttrValImageLayoutType imgLayout = NvSciBufImage_PitchLinearType;
    uint64_t zeroPadding = 0U;
    uint32_t planeWidth = width;
    uint32_t planeHeight = height;
    uint32_t planeBaseAddrAlign = 256U;

    NvSciBufAttrKeyValuePair setAttrs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(NvSciBufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(NvSciBufAttrValAccessPerm) },
        { NvSciBufImageAttrKey_ScanType, &bufScanType, sizeof(bufScanType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &imgCpuAccess, sizeof(bool) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &imgCpuCacheEnabled, sizeof(bool) },
        { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneColorFormat, &planeColorFmt, sizeof(NvSciBufAttrValColorFmt) },
        { NvSciBufImageAttrKey_PlaneColorStd, &planeColorStd, sizeof(NvSciBufAttrValColorStd) },
        { NvSciBufImageAttrKey_Layout, &imgLayout, sizeof(NvSciBufAttrValImageLayoutType) },
        { NvSciBufImageAttrKey_TopPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_BottomPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_LeftPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_RightPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_PlaneWidth, &planeWidth, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneHeight, &planeHeight, sizeof(uint32_t)  },
        { NvSciBufImageAttrKey_PlaneBaseAddrAlign, &planeBaseAddrAlign, sizeof(uint32_t) }
    };
    size_t length = sizeof(setAttrs) / sizeof(NvSciBufAttrKeyValuePair);

    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> unreconciledAttrList;
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> reconciledAttrList;
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> conflictAttrList;
    unreconciledAttrList.reset(new NvSciBufAttrList());
    reconciledAttrList.reset(new NvSciBufAttrList());
    conflictAttrList.reset(new NvSciBufAttrList());
    NvSciError sciErr = NvSciBufAttrListCreate(bufModule, unreconciledAttrList.get());
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate");
    sciErr = NvSciBufAttrListSetAttrs(*unreconciledAttrList, setAttrs, length);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
    NvSciBufAttrList unreconciledAttrLists[2] = { *unreconciledAttrList, bufAttrList };
    sciErr = NvSciBufAttrListReconcile(unreconciledAttrLists,
                                       2U,
                                       reconciledAttrList.get(),
                                       conflictAttrList.get());
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListReconcile");
    sciErr = NvSciBufObjAlloc(*reconciledAttrList, pBufObj);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjAlloc");
    CHK_PTR_AND_RETURN(pBufObj, "NvSciBufObjAlloc");

    return NVSIPL_STATUS_OK;
}

uint8_t * CUtils::CreateImageBuffer(NvSciBufObj bufObj)
{
    BufferAttrs bufAttrs;
    SIPLStatus status = PopulateBufAttr(bufObj, bufAttrs);
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("PopulateBufAttr failed. status: %u\n", status);
        return nullptr;
    }
    uint8_t *buff = new (std::nothrow) uint8_t[bufAttrs.size];
    if (buff == nullptr) {
        LOG_ERR("Failed to allocate memory for image buffer\n");
        return nullptr;
    }
    std::fill(buff, buff + bufAttrs.size, 0x00);

    return buff;
}

bool CUtils::GetBpp(uint32_t buffBits, uint32_t *buffBytesVal) {
    uint32_t buffBytes = 0U;
    if (buffBytesVal == NULL) {
        return false;
    }
    switch(buffBits) {
        case 8:
            buffBytes = 1U;
            break;
        case 10:
        case 12:
        case 14:
        case 16:
            buffBytes = 2U;
            break;
        case 20:
            buffBytes = 3U;
            break;
        case 32:
            buffBytes = 4U;
            break;
        case 64:
            buffBytes = 8U;
            break;
        default:
            LOG_ERR("Invalid planeBitsPerPixels %d\n", buffBits);
            return false;
    }
    *buffBytesVal = buffBytes;
    return true;
}

SIPLStatus CUtils::ConvertRawToRgba(NvSciBufObj srcBufObj,
                                    uint8_t *pSrcBuf,
                                    NvSciBufObj dstBufObj,
                                    uint8_t *pDstBuf)
{
    BufferAttrs srcBufAttrs;
    SIPLStatus status = PopulateBufAttr(srcBufObj, srcBufAttrs);
    CHK_STATUS_AND_RETURN(status, "PopulateBufAttr for source buffer");
    uint8_t *pSrcBufCpy = pSrcBuf;
    uint8_t *pDstBufCpy = pDstBuf;

    NvSciError sciErr = NvSciError_Success;
    if (srcBufAttrs.needSwCacheCoherency) {
        sciErr = NvSciBufObjFlushCpuCacheRange(srcBufObj,
                                               0U,
                                               srcBufAttrs.planePitches[0]
                                                   * srcBufAttrs.planeHeights[0]);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjFlushCpuCacheRange");
    }

    uint32_t bpp = 0U;
    if (!GetBpp(srcBufAttrs.planeBitsPerPixels[0], &bpp)) {
        return NVSIPL_STATUS_ERROR;
    }

    const uint32_t srcPitch = srcBufAttrs.planeWidths[0] * bpp;
    const uint32_t srcBufSize = srcPitch * srcBufAttrs.planeHeights[0];
    sciErr = NvSciBufObjGetPixels(srcBufObj,
                                  nullptr,
                                  (void **)(&pSrcBufCpy),
                                  &srcBufSize,
                                  &srcPitch);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetPixels");
    // Do CPU demosaic
    // Get offsets for each raw component within 2x2 block
    uint32_t xR = 0U, yR = 0U, xG1 = 0U, yG1 = 0U, xG2 = 0U, yG2 = 0U, xB = 0U, yB = 0U;
    switch (srcBufAttrs.planeColorFormats[0]) {
        case NvSciColor_Bayer8RGGB:
        case NvSciColor_Bayer16RGGB:
        case NvSciColor_X2Bayer14RGGB:
        case NvSciColor_X4Bayer12RGGB:
        case NvSciColor_X6Bayer10RGGB:
        case NvSciColor_FloatISP_Bayer16RGGB:
        case NvSciColor_X12Bayer20RGGB:
        case NvSciColor_Bayer16RCCB:
        case NvSciColor_X4Bayer12RCCB:
        case NvSciColor_FloatISP_Bayer16RCCB:
        case NvSciColor_X12Bayer20RCCB:
        case NvSciColor_Bayer16RCCC:
        case NvSciColor_X4Bayer12RCCC:
        case NvSciColor_FloatISP_Bayer16RCCC:
        case NvSciColor_X12Bayer20RCCC:
            xR = 0U; yR = 0U;
            xG1 = 1U; yG1 = 0U;
            xG2 = 0U; yG2 = 1U;
            xB = 1U; yB = 1U;
            break;
        case NvSciColor_Bayer8GRBG:
        case NvSciColor_Bayer16GRBG:
        case NvSciColor_X2Bayer14GRBG:
        case NvSciColor_X4Bayer12GRBG:
        case NvSciColor_X6Bayer10GRBG:
        case NvSciColor_FloatISP_Bayer16GRBG:
        case NvSciColor_X12Bayer20GRBG:
        case NvSciColor_Bayer16CRBC:
        case NvSciColor_X4Bayer12CRBC:
        case NvSciColor_FloatISP_Bayer16CRBC:
        case NvSciColor_X12Bayer20CRBC:
        case NvSciColor_Bayer16CRCC:
        case NvSciColor_X4Bayer12CRCC:
        case NvSciColor_FloatISP_Bayer16CRCC:
        case NvSciColor_X12Bayer20CRCC:
            xG1 = 0U; yG1 = 0U;
            xR = 1U; yR = 0U;
            xB = 0U; yB = 1U;
            xG2 = 1U; yG2 = 1U;
            break;
        case NvSciColor_Bayer8GBRG:
        case NvSciColor_Bayer16GBRG:
        case NvSciColor_X2Bayer14GBRG:
        case NvSciColor_X4Bayer12GBRG:
        case NvSciColor_X6Bayer10GBRG:
        case NvSciColor_FloatISP_Bayer16GBRG:
        case NvSciColor_X12Bayer20GBRG:
        case NvSciColor_Signed_X12Bayer20GBRG:
        case NvSciColor_Bayer16CBRC:
        case NvSciColor_X4Bayer12CBRC:
        case NvSciColor_FloatISP_Bayer16CBRC:
        case NvSciColor_X12Bayer20CBRC:
        case NvSciColor_Bayer16CCRC:
        case NvSciColor_X4Bayer12CCRC:
        case NvSciColor_FloatISP_Bayer16CCRC:
        case NvSciColor_X12Bayer20CCRC:
            xG1 = 0U; yG1 = 0U;
            xB = 1U; yB = 0U;
            xR = 0U; yR = 1U;
            xG2 = 1U; yG2 = 1U;
            break;
        case NvSciColor_Bayer8BGGR:
        case NvSciColor_Bayer16BGGR:
        case NvSciColor_X2Bayer14BGGR:
        case NvSciColor_X4Bayer12BGGR:
        case NvSciColor_X6Bayer10BGGR:
        case NvSciColor_FloatISP_Bayer16BGGR:
        case NvSciColor_X12Bayer20BGGR:
        case NvSciColor_Bayer16BCCR:
        case NvSciColor_X4Bayer12BCCR:
        case NvSciColor_FloatISP_Bayer16BCCR:
        case NvSciColor_X12Bayer20BCCR:
        case NvSciColor_Bayer16CCCR:
        case NvSciColor_X4Bayer12CCCR:
        case NvSciColor_FloatISP_Bayer16CCCR:
        case NvSciColor_X12Bayer20CCCR:
        case NvSciColor_Bayer8CCCC:
        case NvSciColor_Bayer16CCCC:
        case NvSciColor_X2Bayer14CCCC:
        case NvSciColor_X4Bayer12CCCC:
        case NvSciColor_X6Bayer10CCCC:
        case NvSciColor_Signed_X2Bayer14CCCC:
        case NvSciColor_Signed_X4Bayer12CCCC:
        case NvSciColor_Signed_X6Bayer10CCCC:
        case NvSciColor_Signed_Bayer16CCCC:
        case NvSciColor_FloatISP_Bayer16CCCC:
        case NvSciColor_X12Bayer20CCCC:
        case NvSciColor_Signed_X12Bayer20CCCC:
            xB = 0U; yB = 0U;
            xG1 = 1U; yG1 = 0U;
            xG2 = 0U; yG2 = 1U;
            xR = 1U; yR = 1U;
            break;
        default:
            LOG_ERR("Unexpected plane color format\n");
            return NVSIPL_STATUS_ERROR;
    }

    // Demosaic, remembering to skip embedded lines
    for (uint32_t y = srcBufAttrs.topPadding[0];
         y < (srcBufAttrs.planeHeights[0] - static_cast<uint32_t>(srcBufAttrs.bottomPadding[0]));
         y += 2U) {
        for (uint32_t x = 0U; x < srcBufAttrs.planeWidths[0]; x += 2U) {
            // R
            *pDstBuf++ = pSrcBuf[srcPitch*(y + yR) + 2U*(x + xR) + 1U];
            // G (average of G1 and G2)
            uint32_t g1 = pSrcBuf[srcPitch*(y + yG1) + 2U*(x + xG1) + 1U];
            uint32_t g2 = pSrcBuf[srcPitch*(y + yG2) + 2U*(x + xG2) + 1U];
            *pDstBuf++ = (g1 + g2)/2U;
            // B
            *pDstBuf++ = pSrcBuf[srcPitch*(y + yB) + 2U*(x + xB) + 1U];
            // A
            *pDstBuf++ = 0xFF;
        }
    }

    // Write to destination image
    BufferAttrs dstBufAttrs;
    status = PopulateBufAttr(dstBufObj, dstBufAttrs);
    CHK_STATUS_AND_RETURN(status, "PopulateBufAttr for destination buffer");

    if (!GetBpp(dstBufAttrs.planeBitsPerPixels[0], &bpp)) {
        return NVSIPL_STATUS_ERROR;
    }
    const uint32_t dstPitch = dstBufAttrs.planeWidths[0] * bpp;
    const uint32_t dstBufSize = dstPitch * dstBufAttrs.planeHeights[0];
    sciErr = NvSciBufObjPutPixels(dstBufObj,
                                  nullptr,
                                  (const void **)(&pDstBufCpy),
                                  &dstBufSize,
                                  &dstPitch);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjPutPixels");

    return NVSIPL_STATUS_OK;
}

/* Loads NITO file for given camera module.
 The function assumes the .nito files to be named same as camera module name.
 */
SIPLStatus LoadNITOFile(std::string folderPath,
                        std::string moduleName,
                        std::vector<uint8_t>& nito)
{
    // Set up blob file
    string nitoFilePath = (folderPath != "") ? folderPath : NITO_PATH;

    string nitoFile = nitoFilePath + moduleName + ".nito";

    string moduleNameLower{};
    for (auto& c : moduleName) {
        moduleNameLower.push_back(std::tolower(c));
    }
    string nitoFileLower = nitoFilePath + moduleNameLower + ".nito";
    string nitoFileDefault = nitoFilePath + "default.nito";

    // Open NITO file
    auto fp = fopen(nitoFile.c_str(), "rb");
    if (fp == NULL) {
        LOG_INFO("File \"%s\" not found\n", nitoFile.c_str());
        // Try lower case name
        fp = fopen(nitoFileLower.c_str(), "rb");
        if (fp == NULL) {
            LOG_INFO("File \"%s\" not found\n", nitoFileLower.c_str());
            LOG_ERR("Unable to open NITO file for module \"%s\", image quality is not supported!\n", moduleName.c_str());
            return NVSIPL_STATUS_BAD_ARGUMENT;
        } else {
            LOG_MSG("nvsipl_camera: Opened NITO file for module \"%s\", file name: \"%s\" \n",
                    moduleName.c_str(),
                    nitoFileLower.c_str());
        }
    } else {
        LOG_MSG("nvsipl_camera: Opened NITO file for module \"%s\", file name: \"%s\" \n",
                moduleName.c_str(),
                nitoFile.c_str());
    }

    // Check file size
    fseek(fp, 0, SEEK_END);
    auto fsize = ftell(fp);
    rewind(fp);

    if (fsize <= 0U) {
        LOG_ERR("NITO file for module \"%s\" is of invalid size\n", moduleName.c_str());
        fclose(fp);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    /* allocate blob memory */
    nito.resize(fsize);

    /* load nito */
    auto result = (long int) fread(nito.data(), 1, fsize, fp);
    if (result != fsize) {
        LOG_ERR("Fail to read data from NITO file for module \"%s\", image quality is not supported!\n", moduleName.c_str());
        nito.resize(0);
        fclose(fp);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    /* close file */
    fclose(fp);

    LOG_INFO("data from NITO file loaded for module \"%s\"\n", moduleName.c_str());

    return NVSIPL_STATUS_OK;
}

#if !NV_IS_SAFETY
// Fetch NITO Metadata API Utils
SIPLStatus PrintParameterSetID(uint8_t const *const IDArray, size_t const IDArrayLength) {

    if (IDArray == nullptr) {
        LOG_ERR("Parameter Set ID Array is nullptr\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    if (IDArrayLength != 16U) {
        LOG_ERR("Parameter Set ID Array length must be 16U as defined by NITO_PARAMETER_SET_ID_SIZE \n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    cout << setbase(16)
         << setfill('0') << setw(2) << right << hex << unsigned(IDArray[0])
         << setfill('0') << setw(2) << right << hex << unsigned(IDArray[1])
         << setfill('0') << setw(2) << right << hex << unsigned(IDArray[2])
         << setfill('0') << setw(2) << right << hex << unsigned(IDArray[3])
         << "-" << setfill('0') << setw(2) << right << hex << unsigned(IDArray[4])
                << setfill('0') << setw(2) << right << hex << unsigned(IDArray[5])
         << "-" << setfill('0') << setw(2) << right << hex << unsigned(IDArray[6])
                << setfill('0') << setw(2) << right << hex << unsigned(IDArray[7])
         << "-" << setfill('0') << setw(2) << right << hex << unsigned(IDArray[8])
                << setfill('0') << setw(2) << right << hex << unsigned(IDArray[9])
         << setfill('0') << setw(2) << right << hex << unsigned(IDArray[10])
         << setfill('0') << setw(2) << right << hex << unsigned(IDArray[11])
         << setfill('0') << setw(2) << right << hex << unsigned(IDArray[12])
         << setfill('0') << setw(2) << right << hex << unsigned(IDArray[13])
         << setfill('0') << setw(2) << right << hex << unsigned(IDArray[14])
         << setfill('0') << setw(2) << right << hex << unsigned(IDArray[15])
         << endl;

    return NVSIPL_STATUS_OK;
}

SIPLStatus PrintParameterSetSchemaHash(uint8_t const *const schemaHashArray, size_t const schemaHashArrayLength) {
    if (schemaHashArray == nullptr) {
        LOG_ERR("Schema Hash Array is nullptr\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    if (schemaHashArrayLength != 32U) {
        LOG_ERR("Schema Hash Array Array length must be 32U as defined by NITO_SCHEMA_HASH_SIZE \n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    for (auto i = 0U; i < schemaHashArrayLength; ++i)
    {
        cout << setfill('0') << setw(2) << right << hex
             << unsigned(schemaHashArray[i]);
    }
    cout << "\n";
    return NVSIPL_STATUS_OK;
}

SIPLStatus PrintParameterSetDataHash(uint8_t const *const dataHashArray, size_t const dataHashArrayLength) {
    if (dataHashArray == nullptr) {
        LOG_ERR("Data Hash Array is nullptr\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    if (dataHashArrayLength != 32U) {
        LOG_ERR("Data Hash Array length must be 32U as defined by NITO_DATA_HASH_SIZE \n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    for (auto i = 0U; i < dataHashArrayLength; ++i)
    {
        cout << setfill('0') << setw(2) << right << hex
             << unsigned(dataHashArray[i]);
    }
    cout << "\n";
    return NVSIPL_STATUS_OK;
}
#endif // !NV_IS_SAFETY

SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs &bufAttrs)
{
    NvSciError err = NvSciError_Success;
    NvSciBufAttrList bufAttrList;

    NvSciBufAttrKeyValuePair imgAttrs[] = {
        { NvSciBufImageAttrKey_Size, NULL, 0 },               //0
        { NvSciBufImageAttrKey_Layout, NULL, 0 },             //1
        { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },         //2
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },         //3
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },        //4
        { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },         //5
        { NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0 },  //6
        { NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0 }, //7
        { NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0 },   //8
        { NvSciBufImageAttrKey_PlaneChannelCount, NULL, 0 },  //9
        { NvSciBufImageAttrKey_PlaneOffset, NULL, 0 },        //10
        { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },    //11
        { NvSciBufImageAttrKey_TopPadding, NULL, 0 },        //12
        { NvSciBufImageAttrKey_BottomPadding, NULL, 0 },   //13
        { NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0 } //14
    };

    err = NvSciBufObjGetAttrList(sciBufObj, &bufAttrList);
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufObjGetAttrList");
    err = NvSciBufAttrListGetAttrs(bufAttrList, imgAttrs, sizeof(imgAttrs) / sizeof(imgAttrs[0]));
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListGetAttrs");

    bufAttrs.size = *(static_cast<const uint64_t*>(imgAttrs[0].value));
    bufAttrs.layout = *(static_cast<const NvSciBufAttrValImageLayoutType*>(imgAttrs[1].value));
    bufAttrs.planeCount = *(static_cast<const uint32_t*>(imgAttrs[2].value));
    bufAttrs.needSwCacheCoherency = *(static_cast<const bool*>(imgAttrs[14].value));

    memcpy(bufAttrs.planeWidths,
        static_cast<const uint32_t*>(imgAttrs[3].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeHeights,
        static_cast<const uint32_t*>(imgAttrs[4].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planePitches,
        static_cast<const uint32_t*>(imgAttrs[5].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeBitsPerPixels,
        static_cast<const uint32_t*>(imgAttrs[6].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeAlignedHeights,
        static_cast<const uint32_t*>(imgAttrs[7].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeAlignedSizes,
        static_cast<const uint64_t*>(imgAttrs[8].value),
        bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeChannelCounts,
        static_cast<const uint8_t*>(imgAttrs[9].value),
        bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeOffsets,
        static_cast<const uint64_t*>(imgAttrs[10].value),
        bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeColorFormats,
        static_cast<const NvSciBufAttrValColorFmt*>(imgAttrs[11].value),
        bufAttrs.planeCount * sizeof(NvSciBufAttrValColorFmt));
    memcpy(bufAttrs.topPadding,
        static_cast<const uint32_t*>(imgAttrs[12].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.bottomPadding,
        static_cast<const uint32_t*>(imgAttrs[13].value),
        bufAttrs.planeCount * sizeof(uint32_t));

    return NVSIPL_STATUS_OK;
}
