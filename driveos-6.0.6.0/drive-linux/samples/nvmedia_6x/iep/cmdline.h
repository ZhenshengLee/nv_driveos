/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _NVMEDIA_TEST_CMD_LINE_H_
#define _NVMEDIA_TEST_CMD_LINE_H_

#include <stdbool.h>

#include "nvmedia_common_encode.h"

#define DEFAULT_GOP_SIZE                30
#define DEFAULT_FRAME_SIZE              15000
#define ARRAYS_ALOCATION_SIZE           1000
#define FILE_NAME_SIZE                  256
#define FILE_NAME_SIZE_EXTRA            (FILE_NAME_SIZE+32)
#define MAX_CONFIG_SECTIONS             32
#define INTERVAL_PATTERN_MAX_LENGTH     1000
#define MAX_PAYLOAD_ARRAY_SIZE          10
#define FILE_PATH_LENGTH_MAX            256
#define IMAGE_BUFFERS_POOL_SIZE         5
#define IMAGE_BUFFERS_POOL_SIZE_MAX     16
#define IMAGE_BUFFERS_POOL_SIZE_MIN     1

#if defined(__QNX__)
// Cycle count measurement can be converted to nanosecond by multiplication with 32.
#define CYCLECOUNT_TO_NS                32
#endif

typedef struct {
    char        crcFilename[FILE_PATH_LENGTH_MAX];
    bool crcGenMode;
    bool crcCheckMode;
} CRCOptions;

typedef struct _EncodePicParams {
    unsigned int            encodePicFlags;
    unsigned long long      inputDuration;
    NvMediaEncodePicType    pictureType;
    unsigned int            PicParamsSectionNum;
    unsigned int            rcParamsSectionNum;
} EncodePicParams;

typedef struct _EncodePicParamsH264 {
    bool     refPicFlag;
    unsigned int    forceIntraRefreshWithFrameCnt;
    unsigned char   sliceTypeData[ARRAYS_ALOCATION_SIZE];
    unsigned int    sliceTypeArrayCnt;
    char            payloadArrayIndexes[MAX_PAYLOAD_ARRAY_SIZE];
    unsigned int    payloadArraySize;
    unsigned int    mvcPicParamsSectionNum;
} EncodePicParamsH264;

typedef struct _EncodeH264SEIPayload {
    unsigned int    payloadSize;
    unsigned int    payloadType;
    unsigned char   payload[ARRAYS_ALOCATION_SIZE];
} EncodeH264SEIPayload;

typedef struct _EncodePicParamsH265 {
    bool     refPicFlag;
    unsigned int    forceIntraRefreshWithFrameCnt;
    unsigned char   sliceTypeData[ARRAYS_ALOCATION_SIZE];
    unsigned int    sliceTypeArrayCnt;
    char            payloadArrayIndexes[MAX_PAYLOAD_ARRAY_SIZE];
    unsigned int    payloadArraySize;
} EncodePicParamsH265;

typedef struct _EncodeH265SEIPayload {
    unsigned int    payloadSize;
    unsigned int    payloadType;
    unsigned char   payload[ARRAYS_ALOCATION_SIZE];
} EncodeH265SEIPayload;

typedef struct _EncodeConfig {
    unsigned char   profile;
    unsigned char   level;
    int             gopPattern;
    unsigned int    gopLength;
    unsigned int    encodeWidth;
    unsigned int    encodeHeight;
    unsigned int    darWidth;
    unsigned int    darHeight;
    unsigned int    frameRateNum;
    unsigned int    frameRateDen;
    unsigned char   maxNumRefFrames;
    bool            useBFramesAsRef;
    bool            enableAllIFrames;
    bool            enableSsimRdo;
    bool            enableTileEncode;
    unsigned int    log2NumTilesInRow;
    unsigned int    log2NumTilesInCol;
    unsigned int    vp9SkipChroma;
    unsigned int    frameRestorationType;
    bool            enableBiCompound;
    bool            enableUniCompound;
    unsigned int    numEpCores;
    bool            ampDisable;
} EncodeConfig;

typedef struct _EncodeRCParams {
    NvMediaEncodeParamsRCMode   rcMode;
    unsigned int                rcConstQPSectionNum;
    unsigned int                averageBitRate;
    unsigned int                maxBitRate;
    unsigned int                vbvBufferSize;
    unsigned int                vbvInitialDelay;
    bool                        enableMinQP;
    bool                        enableMaxQP;
    unsigned int                rcMinQPSectionNum;
    unsigned int                rcMaxQPSectionNum;
} EncodeRCParams;

typedef struct _TestArgs {
    char                        infile[FILE_NAME_SIZE];
    char                        outfile[FILE_NAME_SIZE];
#if !NV_IS_SAFETY
    char                        mvDataFileName[FILE_NAME_SIZE];
    char                        extradataFileName[FILE_NAME_SIZE];
    char                        dynResFileName[FILE_NAME_SIZE];
    char                        qpDeltaFileBaseName[FILE_NAME_SIZE];
    char                        infiledrc[FILE_NAME_SIZE];
    char                        dynBitrateFileName[FILE_NAME_SIZE];
    char                        dynFpsFileName[FILE_NAME_SIZE];
    char                        fslFileName[FILE_NAME_SIZE];
    unsigned int                dynResFrameNum;
    unsigned int                dynResFrameWidth;
    unsigned int                dynResFrameHeight;
    unsigned int                qpDeltaMapBufferEnabled;
    unsigned int                drcBufRealloc;
#endif
    unsigned int                inputFileFormat;
    unsigned int                startFrame;
    unsigned int                framesToBeEncoded;
    unsigned int                videoCodec;
    unsigned int                maxOutputBuffering;
    unsigned int                rateControlSectionNum;
    char                        frameIntervalPattern[INTERVAL_PATTERN_MAX_LENGTH];
    unsigned int                frameIntervalPatternLength;
    EncodeConfig                configParams;
    NvMediaEncodeConfigH264     configH264Params;
    EncodePicParams             picParamsCollection[MAX_CONFIG_SECTIONS];
    EncodePicParamsH264         picH264ParamsCollection[MAX_CONFIG_SECTIONS];
    EncodeRCParams              rcParamsCollection[MAX_CONFIG_SECTIONS];
    EncodeH264SEIPayload        payloadsCollection[MAX_CONFIG_SECTIONS];
    NvMediaEncodeQP             quantizationParamsCollection[MAX_CONFIG_SECTIONS];

    NvMediaEncodeConfigH265     configH265Params;
    EncodePicParamsH265         picH265ParamsCollection[MAX_CONFIG_SECTIONS];
    EncodeH265SEIPayload        payloadsH265Collection[MAX_CONFIG_SECTIONS];

    NvMediaEncodeConfigVP9      configVP9Params;
    NvMediaEncodeConfigAV1      configAV1Params;
    unsigned long long int      sumCycleCount;

    CRCOptions                  crcoption;
    bool                        eventDataRecorderMode;
    int                         eventDataRecorderRecordingTime;
    int                         logLevel;

    unsigned int                instanceId;

    bool                        perfTest;
    double                      avgTimePerFrame;
    bool                        negativeTest;
    bool                        skipImageRegister;
    bool                        version;
    bool                        alternateCreateAPI;
    bool                        enableInternalHighBitDepth;
#if !NV_IS_SAFETY
    bool                        enableExtradata;
    bool                        dumpFrameSizeLog;
    unsigned char               dumpFslLevel;
#endif
#if ENABLE_PROFILING
    int                         profileEnable;
    int                         limitFPS;
    char                        profileStatsFilePath[FILE_NAME_SIZE];
#endif
    uint64_t                    loopCount;
} TestArgs;

void PrintUsage(void);
int  ParseArgs(int argc, char **argv, TestArgs *args);

#endif /* _NVMEDIA_TEST_CMD_LINE_H_ */
