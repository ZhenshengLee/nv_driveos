/*
 * Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>

#include "log_utils.h"
#include "misc_utils.h"

#define  CRC32_POLYNOMIAL   0xEDB88320L

uint32_t
u32(const uint8_t* ptr)
{
    return ptr[0] | (ptr[1]<<8) | (ptr[2]<<16) | (ptr[3]<<24);
}

NvMediaStatus
GetTimeMicroSec(
    uint64_t *uTime)
{
    struct timespec t;
#if !(defined(CLOCK_MONOTONIC) && defined(_POSIX_MONOTONIC_CLOCK) && _POSIX_MONOTONIC_CLOCK >= 0 && _POSIX_TIMERS > 0)
    struct timeval tv;
#endif

    if(!uTime)
        return NVMEDIA_STATUS_BAD_PARAMETER;

#if !(defined(CLOCK_MONOTONIC) && defined(_POSIX_MONOTONIC_CLOCK) && _POSIX_MONOTONIC_CLOCK >= 0 && _POSIX_TIMERS > 0)
    gettimeofday(&tv, NULL);
    t.tv_sec = tv.tv_sec;
    t.tv_nsec = tv.tv_usec*1000L;
#else
    clock_gettime(CLOCK_MONOTONIC, &t);
#endif

    *uTime = (uint64_t)t.tv_sec * 1000000LL + (uint64_t)t.tv_nsec / 1000LL;
    return NVMEDIA_STATUS_OK;
}

static void
BuildCRCTable(
    uint32_t *crcTable)
{
    uint16_t i;
    uint16_t j;
    uint32_t crc;

    if (!crcTable) {
        LOG_ERR("BuildCRCTable: Failed creating CRC table - bad pointer for crcTable %p\n", crcTable);
        return;
    }

    for (i = 0; i <= 255; i++) {
        crc = i;
        for (j = 8; j > 0; j--) {
            if (crc & 1) {
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc >>= 1;
            }
        }
        crcTable[i] = crc;
    }
    return;
}

uint32_t
CalculateBufferCRC(
    uint32_t count,
    uint32_t crc,
    uint8_t *buffer)
{
    uint8_t *p;
    uint32_t temp1;
    uint32_t temp2;
    static uint32_t crcTable[256];
    static int initialized = 0;

    if(!initialized) {
        BuildCRCTable(crcTable);
        initialized = 1;
    }
    p = (uint8_t*) buffer;
    while (count-- != 0) {
        temp1 = (crc >> 8) & 0x00FFFFFFL;
        temp2 = crcTable[((uint32_t) crc ^ *p++) & 0xFF];
        crc = temp1 ^ temp2;
    }
    return crc;
}

int32_t
ParseRCVHeader(
    RCVFileHeader *pHdr,
    const uint8_t *pBuffer,
    int32_t lBufferSize)
{
    int32_t lNumFrames, profile, level;
    uint32_t uType, tmp, res1, uHdrSize;
    uint32_t cur = 0;

    // The first 3 bytes are the number of frames
    lNumFrames = pBuffer[cur++];
    lNumFrames |= pBuffer[cur++] << 8;
    lNumFrames |= pBuffer[cur++] << 16;
    if (lNumFrames <= 0)
        return 0;
    pHdr->lNumFrames = lNumFrames;
    LOG_DBG("pHdr->lNumFrames = %d \n", pHdr->lNumFrames);
    // The next byte is the type and extension flag
    uType = pBuffer[cur++];
    LOG_DBG("uType = %d \n", uType);
    if ((uType & ~RCV_V2_MASK) != RCV_VC1_TYPE)
        return 0;
    pHdr->bRCVIsV2Format = ((uType & RCV_V2_MASK) != 0);
    LOG_DBG("pHdr->bRCVIsV2Format = %d \n", pHdr->bRCVIsV2Format);
    // Next 4 bytes are the size of the extension data
    pHdr->cbSeqHdr = u32(pBuffer+cur);
    LOG_DBG("pHdr->cbSeqHdr = %d \n", pHdr->cbSeqHdr);
    cur += 4;
    memcpy(pHdr->SeqHdrData, pBuffer+cur, pHdr->cbSeqHdr);
    // STRUCT_C
    profile = pBuffer[cur] >> 6;
    cur += pHdr->cbSeqHdr;
    LOG_DBG("VC1 profile = %d \n", profile);
    if (profile >= 2) {
        LOG_ERR("High profile RCV is not supported\n");
        return 0;   // Must be Simple or Main (AP handled as VC1 elementary stream)
    }
    // STRUCT_A
    pHdr->lMaxCodedHeight = u32(pBuffer+cur);
    LOG_DBG("pHdr->lMaxCodedHeight = %d \n", pHdr->lMaxCodedHeight);
    cur += 4;
    if ((pHdr->lMaxCodedHeight <= 31) || (pHdr->lMaxCodedHeight > 2048-32))
        return 0;
    pHdr->lMaxCodedWidth = u32(pBuffer+cur);
    LOG_DBG("pHdr->lMaxCodedWidth = %d \n", pHdr->lMaxCodedWidth);
    cur += 4;
    if ((pHdr->lMaxCodedWidth <= 15) || (pHdr->lMaxCodedWidth > 4096-16))
        return 0;
    tmp = u32(pBuffer+cur); // 0x0000000c
    cur += 4;
    if (tmp != 0x0000000c)
        return 0;
    // STRUCT_B
    tmp = u32(pBuffer+cur);  // level = tmp >> 29 & 0x7; cbr = tmp >> 28 & 0x1;
    cur += 4;
    level = tmp >> 29;
    res1 = (tmp >> 24) & 0xf;
    if ((res1 != 0x0) || (level > 0x4))
        return 0;
    pHdr->lHrdBuffer = (tmp >> 0) & 0xffffff;
    tmp = u32(pBuffer+cur);
    cur += 4;
    pHdr->lBitRate = (tmp >> 0) & 0xffffff;
    pHdr->lFrameRate = u32(pBuffer+cur);
    cur += 4;
    uHdrSize = cur;
    LOG_DBG("uHdrSize = %d \n", uHdrSize);
    return uHdrSize;
}

NvMediaStatus
SetRect (
    NvMediaRect *rect,
    unsigned short x0,
    unsigned short y0,
    unsigned short x1,
    unsigned short y1)
{
    if(!rect)
        return NVMEDIA_STATUS_BAD_PARAMETER;

    rect->x0 = x0;
    rect->x1 = x1;
    rect->y0 = y0;
    rect->y1 = y1;

    return NVMEDIA_STATUS_OK;
}

uint8_t *
readFileToMemory(char *fileName, uint64_t *fileSize)
{
    uint8_t *memoryPtr = NULL;
    FILE *fp = NULL;
    uint32_t elementsRead;
    int64_t fileLength;

    if((fileName == NULL) || (fileSize == NULL)) {
        LOG_ERR("Bad parameter.\n");
        goto readFileToMemoryEnd;
    }

    fp = fopen(fileName, "r");
    if(fp == NULL) {
        LOG_ERR("Unable to open file.\n");
        goto readFileToMemoryEnd;
    }
    fseek(fp, 0L, SEEK_END);
    fileLength = ftell(fp);
    if(fileLength < 0) {
        LOG_ERR("Unable to get size of file.\n");
        goto readFileToMemoryEnd;
    }
    rewind(fp);

    memoryPtr = (uint8_t *) malloc(fileLength);
    if(memoryPtr == NULL) {
        LOG_ERR("Unable to allocate memory for file.\n");
        goto readFileToMemoryEnd;
    }
    elementsRead = fread(memoryPtr, fileLength, 1, fp);
    if(elementsRead != 1) {
        LOG_ERR("Error reading from file.\n");
        free(memoryPtr);
        memoryPtr = NULL;
        goto readFileToMemoryEnd;
    }
    *fileSize = fileLength;

readFileToMemoryEnd:
    if(fp) {
        fclose(fp);
    }
    return memoryPtr;
}

NvMediaStatus
compareFiles(char *fileName1, char* fileName2, bool *compareResult){
    uint8_t *f1p = NULL, *f2p = NULL;
    uint64_t f1Size, f2Size;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;

    if((fileName1 == NULL) || (fileName2 == NULL) || (compareResult == NULL)) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        LOG_ERR("%s: Bad parameter.\n", __func__);
        goto compareFilesEnd;
    }

    *compareResult = false; //set to False by default

    f1p = readFileToMemory(fileName1, &f1Size);
    if(f1p == NULL) {
        LOG_ERR("%s: Error reading file1 to memory.\n", __func__);
        status = NVMEDIA_STATUS_ERROR;
        goto compareFilesEnd;
    }

    f2p = readFileToMemory(fileName2, &f2Size);
    if(f2p == NULL) {
        LOG_ERR("%s: Error reading file2 to memory.\n", __func__);
        status = NVMEDIA_STATUS_ERROR;
        goto compareFilesEnd;
    }

    if ((f1Size == f2Size) && memcmp(f1p, f2p, f1Size) == 0){
        *compareResult = true;
    }

    status = NVMEDIA_STATUS_OK;

compareFilesEnd:
    if (f1p){
        free(f1p);
    }
    if (f2p){
        free(f2p);
    }

    return status;
}
