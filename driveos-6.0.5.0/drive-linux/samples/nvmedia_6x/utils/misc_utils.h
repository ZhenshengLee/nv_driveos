/*
 * Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _NVMEDIA_TEST_MISC_UTILS_H_
#define _NVMEDIA_TEST_MISC_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "nvmedia_core.h"

#define RCV_MAX_FRAME_SIZE   2048 * 1024
#define RCV_VC1_TYPE         0x85
#define RCV_V2_MASK          (1<<6)
#ifndef __INTEGRITY
#define MIN(a,b)             (((a) < (b)) ? (a) : (b))
#define MAX(a,b)             (((a) > (b)) ? (a) : (b))
#endif
#define COPYFIELD(a,b,field) (a)->field = (b)->field
#define MAX_STRING_SIZE      256
#define MAX_OUTPUT_DEVICES   4

#define IsFailed(result)    result != NVMEDIA_STATUS_OK
#define IsSucceed(result)   result == NVMEDIA_STATUS_OK

typedef enum {
    LSB_ALIGNED,
    MSB_ALIGNED
} PixelAlignment;

typedef struct _RCVFileHeader {
    int32_t lNumFrames;
    int32_t bRCVIsV2Format;
    uint32_t   uProfile;
    int32_t lMaxCodedWidth;
    int32_t lMaxCodedHeight;
    int32_t lHrdBuffer;
    int32_t lBitRate;
    int32_t lFrameRate;
    int32_t cbSeqHdr;       // Should always be 4 for simple/main
    uint8_t SeqHdrData[32];
} RCVFileHeader;

//  u32
//
//    u32()  Reads 4 bytes from buffer and returns the read value
//
//  Arguments:
//
//   ptr
//      (in) Input buffer

uint32_t   u32(const uint8_t* ptr);

//  GetTimeMicroSec
//
//    GetTimeMicroSec()  Returns current time in microseconds
//
//  Arguments:
//
//   uTime
//      (out) Pointer to current time in microseconds

NvMediaStatus
GetTimeMicroSec(
    uint64_t *uTime);

//  CalculateBufferCRC
//
//    CalculateBufferCRC()  Calculated CRC for a given buffer and base CRC value
//
//  Arguments:
//
//   count
//      (in) buffer length in bytes
//
//   crc
//      (in) Base CRC value
//
//   buffer
//      (in) Pointer to buffer

uint32_t
CalculateBufferCRC(
    uint32_t count,
    uint32_t crc,
    uint8_t *buffer);

int32_t
ParseRCVHeader(
    RCVFileHeader *pHdr,
    const uint8_t *pBuffer,
    int32_t lBufferSize);

//  SetRect
//
//    SetRect()  Sets NvMediaRect structure with given values
//
//  Arguments:
//
//   rect
//      (in) Pointer to NvMediaRect
//
//   x0
//      (in) x0 point of the rectangle
//
//   y0
//      (in) y0 point of the rectangle
//
//   x1
//      (in) x1 point of the rectangle
//
//   y1
//      (in) y1 point of the rectangle

NvMediaStatus
SetRect (
    NvMediaRect *rect,
    unsigned short x0,
    unsigned short y0,
    unsigned short x1,
    unsigned short y1);

//   readFileToMemory
//
//      readFileToMemory()     allocates memory and reads file into memory.
//
//   Returns pointer to memory. Return NULL in case of any error
//
//   Caller is responsible for freeing memory.
//
//   Argument:
//
//      filename
//          (in) String containing the name of the first file to be compared
//
//      filesize
//          (out) Pointer to file size

uint8_t *
readFileToMemory(char *fileName, uint64_t *fileSize);

//   compareFiles
//
//   compareFiles()     Compare two files for its content
//
//   NOTE: Comparison result is set to false by default
//
//   Argument:
//
//      filename1
//          (in) String containing the name of the first file to be compared
//
//      filename2
//          (in) String containing the name of the second file to be compared
//
//      compareResult
//          (out) Result of comparison will either be one of the following:
//                false - Default result or when file size or file data
//                                do not match
//                true - File size and file data match
//
//      NvMediaStatus
//          (out) Returns status of function.
//                NVMEDIA_STATUS_OK on successful execution
//                NVMEDIA_STATUS_BAD_PARAMETER on passing invalid argument(s)
//                NVMEDIA_STATUS_ERROR in case of any other error

NvMediaStatus
compareFiles(char *fileName1, char* fileName2, bool *compareResult);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_MISC_UTILS_H_ */
