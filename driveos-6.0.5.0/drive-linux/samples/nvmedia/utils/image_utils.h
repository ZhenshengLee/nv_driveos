/*
 * Copyright (c) 2019-2020 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _NVMEDIA_TEST_IMAGE_UTILS_H_
#define _NVMEDIA_TEST_IMAGE_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "misc_utils.h"
#include "nvmedia_core.h"
#include "nvmedia_surface.h"
#include "nvmedia_image.h"

#if (NV_IS_SAFETY == 1)
#include "nvmedia_image_internal.h"
#endif

#define PACK_RGBA(R, G, B, A)  (((uint32_t)(A) << 24) | ((uint32_t)(B) << 16) | \
                                ((uint32_t)(G) << 8) | (uint32_t)(R))
#define DEFAULT_ALPHA   0x80


typedef struct {
    unsigned char  *pSurf;
    unsigned int    width;
    unsigned int    height;
    unsigned int    pitch;
    unsigned int    bpp;
} MemSurf;

//  ReadRGBAFile
//
//    ReadRGBAFile()  Read surface from RGBA file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   width
//      (in) Surface width
//
//   height
//      (in) Surface height
//
//   rgbaSurface
//      (out) Pointer to pre-allocated MemSurf structure

NvMediaStatus
ReadRGBA(
    char *filename,
    unsigned int width,
    unsigned int height,
    MemSurf *rgbaSurface);

//  WriteRGBA
//
//    WriteRGBA()  Write surface to an RGBA (binary) file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   outputBpp
//      (in) Output bytes per pixel
//
//   defaultAplha
//      (in) Default alpha in case of an RGB file
//
//   rgbaSurface
//      (in) Pointer to pre-allocated MemSurf structure

NvMediaStatus
WriteRGBA(
    char *filename,
    uint32_t outputBpp,
    uint8_t defaultAplha,
    MemSurf *rgbaSurface);

//  GetPPMFileDimensions
//
//    GetPPMFileDimensions() Gets surface dimenssions from a PPM file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   uWidth
//      (out) Pointer to surface width
//
//   uHeight
//      (out) Pointer to surface height

NvMediaStatus
GetPPMFileDimensions(
    char *fileName,
    uint16_t *uWidth,
    uint16_t *uHeight);

//  ReadPPM
//
//    ReadPPM()  Read PPM file to a surface
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   rgbaSurface
//      (out) Pointer to pre-allocated surface

NvMediaStatus
ReadPPM(
    char *fileName,
    uint8_t defaultAplha,
    MemSurf *rgbaSurface);

//  WritePPM
//
//    WritePPM()  Write a surface to PPM file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   rgbaSurface
//      (in) Pointer to pre-allocated surface

NvMediaStatus
WritePPM(
    char *fileName,
    MemSurf *rgbaSurface);

//  ReadPAL
//
//    ReadPAL()  Read binary palette from file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   palette
//      (out) Pointer to pre-allocated palette

NvMediaStatus
ReadPAL(
    char *filename,
    uint32_t *palette);

//  ReadI8
//
//    ReadI8()  Read I8 (indexed) file to a surface
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   surface
//      (out) Pointer to pre-allocated surface

NvMediaStatus
ReadI8(
    char *filename,
    MemSurf *surface);

//  CreateMemRGBASurf
//
//    CreateMemRGBASurf()  Creates RGBA surface and initializes the values (optional)
//
//  Arguments:
//
//   width
//      (in) Surface width
//
//   height
//      (in) Surface height
//
//   initflag
//      (in) Flag for surface initialization
//
//   initvalue
//      (in) Init value. Surface will be initialized with this value if initflag is true
//
//   surf_out
//      (out) Pointer to pointer to created surface

NvMediaStatus
CreateMemRGBASurf(
    unsigned int width,
    unsigned int height,
    NvMediaBool initflag,
    unsigned int initvalue,
    MemSurf **surf_out);

//  DestroyMemSurf
//
//    DestroyMemSurf()  Releasing surface memory
//
//  Arguments:
//
//   surf
//      (in) Pointer to released surface

NvMediaStatus
DestroyMemSurf(
    MemSurf *surf);

//  DrawRGBARect
//
//    DrawRGBARect()  Creates RGBA rectangle with chosen color
//
//  Arguments:
//
//   surf
//      (out) Pointer to pre-allocated output surface
//
//   rect
//      (in) Pointer to requested rectangle structure
//
//   R
//      (in) R value
//
//   G
//      (in) G value
//
//   B
//      (out) B value
//
//   A
//      (out) A value

NvMediaStatus
DrawRGBARect(
    MemSurf *surf,
    NvMediaRect *rect,
    uint8_t R,
    uint8_t G,
    uint8_t B,
    uint8_t A);

//  PreMultiplyRGBASurf
//
//    PreMultiplyRGBASurf()  Multiplies RGBA surface
//
//  Arguments:
//
//   surf
//      (in/out) Pointer to pre-allocated surface

NvMediaStatus
PreMultiplyRGBASurf(
    MemSurf *surf);

//  CreateMemI8Surf
//
//    CreateMemI8Surf()  Creates and initializes I8 surface
//
//  Arguments:
//
//   width
//      (in) Surface width
//
//   height
//      (in) Surface height
//
//   init
//      (in) Init value for surface initialization
//
//   surf_out
//      (out) Pointer to output surface

NvMediaStatus
CreateMemI8Surf(
    uint32_t width,
    uint32_t height,
    uint8_t init,
    MemSurf **surf_out);

//  DrawI8Rect
//
//    DrawI8Rect()  Creates and initializing I8 rectangle
//
//  Arguments:
//
//   surf
//      (out) Pointer to pre-allocated output surface
//
//   rect
//      (in) Pointer to requested rectangle structure
//
//   index
//      (in) Initialization  value

NvMediaStatus
DrawI8Rect(
    MemSurf *surf,
    NvMediaRect *rect,
    uint8_t index);

NvMediaStatus
GetImageCrc(
    NvMediaImage *image,
    uint32_t width,
    uint32_t height,
    uint32_t *crcOut,
    uint32_t rawBytesPerPixel);

NvMediaStatus
CheckImageCrc(
    NvMediaImage *image,
    uint32_t width,
    uint32_t height,
    uint32_t ref,
    NvMediaBool *isMatching,
    uint32_t rawBytesPerPixel);

NvMediaStatus
CheckImageOutput(
    NvMediaImage *image,
    uint32_t width,
    uint32_t height,
    uint8_t* SrcBuf,
    NvMediaBool *isMatching,
    uint32_t rawBytesPerPixel);


//  WriteImage
//
//    WriteImage()  Save RGB or YUV image to a file
//
//  Arguments:
//
//   filename
//      (in) Output file name
//
//   image
//      (out) Pointer to the image
//
//   uvOrderFlag
//      (in) Flag for UV order. If true - UV; If false - VU;
//           Used only in YUV type surface case
//
//   appendFlag
//      (in) Apped to exisitng file if true otherwise create new file
//
//   bytesPerPixel
//      (in) Bytes per pixel. Nedded for RAW image types handling.
//         RAW8 - 1 byte per pixel
//         RAW10, RAW12, RAW14 - 2 bytes per pixel
//
//   srcRect
//      (in) structure containing co-ordinates of the rectangle in the source surface
//           from which the client surface is to be copied. Setting srcRect to NULL
//           implies rectangle of full surface size.

NvMediaStatus
WriteImage(
    char *filename,
    NvMediaImage *image,
    NvMediaBool uvOrderFlag,
    NvMediaBool appendFlag,
    uint32_t bytesPerPixel,
    NvMediaRect *srcRect);

//  ReadImage
//
//    ReadImage()  Read image from file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   frameNum
//      (in) Frame number to read. Use for stream input files.
//
//   width
//      (in) Surface width
//
//   height
//      (in) Surface height
//
//   image
//      (out) Pointer to pre-allocated output surface
//
//   uvOrderFlag
//      (in) Flag for UV order. If true - UV; If false - VU;
//
//   bytesPerPixel
//      (in) Bytes per pixel. Nedded for RAW image types handling.
//         RAW8 - 1 byte per pixel
//         RAW10, RAW12, RAW14 - 2 bytes per pixel
//
//   pixelAlignment
//      (in) Alignment of bits in pixel.
//         0 - LSB Aligned
//         1 - MSB Aligned

NvMediaStatus
ReadImage(
    char *fileName,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    NvMediaImage *image,
    NvMediaBool uvOrderFlag,
    uint32_t bytesPerPixel,
    uint32_t pixelAlignment);

//  WriteRAWImageToRGBA
//
//    WriteRAWImageToRGBA()  Converts RAW image to RGBA and saves to file
//
//  Arguments:
//
//   filename
//      (in) Output file name
//
//   image
//      (in) Pointer to RAW image
//
//   appendFlag
//      (in) Apped to exisitng file if true otherwise create new file
//
//   bytesPerPixel
//      (in) Bytes per pixel.
//         RAW8 - 1 byte per pixel
//         RAW10, RAW12, RAW14 - 2 bytes per pixel

NvMediaStatus
WriteRAWImageToRGBA(
    char *filename,
    NvMediaImage *image,
    NvMediaBool appendFlag,
    uint32_t bytesPerPixel);

//  ReadPPMImage
//
//    ReadPPMImage()  Read PPM file to a image surface
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   image
//      (out) Pointer to pre-allocated image surface

NvMediaStatus
ReadPPMImage(
    char *fileName,
    NvMediaImage *image);

//  InitImage
//
//    InitImage()  Init image data to zeros
//
//  Arguments:
//
//   image
//      (in) image to initialize
//
//   width
//      (in) Surface width
//
//   height
//      (in) Surface height

NvMediaStatus
InitImage(
    NvMediaImage *image,
    uint32_t width,
    uint32_t height);

//  ReadYUVBuffer
//
//    ReadYUVBuffer()  Read specific frame from YUV file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   uFrameNum
//      (in) Frame number to read
//
//   width
//      (in) buffer width
//
//   height
//      (in) buffer height
//
//   pFrame
//      (out) Pointer to pre-allocated output buffer
//
//   bOrderUV
//      (in) Flag for UV order. If true - UV; If false - VU;
NvMediaStatus
ReadYUVBuffer(
    FILE *file,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    uint8_t *pBuff,
    NvMediaBool bOrderUV);

//  ReadRGBABuffer
//
//    ReadRGBABuffer()  Read buffer from RGBA file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   width
//      (in) buffer width
//
//   height
//      (in) buffer height
//
//   rgbaSurface
//      (out) Pointer to pre-allocated output buffer

NvMediaStatus
ReadRGBABuffer(
    FILE *file,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    uint8_t *pBuff);

NvMediaStatus
GetFrameCrc(
    uint8_t   **pBuff,
    uint32_t  *widths,
    uint32_t  *heights,
    uint32_t  *pitches,
    uint32_t  numSurfaces,
    uint32_t  *crcOut);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_IMAGE_UTILS_H_ */
