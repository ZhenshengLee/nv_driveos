/* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include <iostream>

#include "common_defs.h"
#include "ofa_class.h"
#include "image_buffer.h"
#include "file_writer.h"
#include "image_pyramid.h"

using namespace std;

bool RegisterSurfaces(NvMOfa *pNvMOfa, Buffer *image1, Buffer *image2, Buffer *outSurface,Buffer *costSurface);
bool RegisterSurfaces(NvMOfa *pNvMOfa, Buffer *outSurface,Buffer *costSurface);
void UnRegisterSurfaces(NvMOfa *pNvMOfa, Buffer *image1, Buffer *image2, Buffer *outSurface, Buffer *costSurface);
void UnRegisterSurfaces(NvMOfa *pNvMOfa, Buffer *outSurface, Buffer *costSurface);
bool writeBinFile(string filename, char *surface, uint32_t size);
bool RegisterSurfaces(NvMOfaFlow *pNvMOfa, ImagePyramid *image1, ImagePyramid *image2, ImagePyramid *outSurface, ImagePyramid *costSurface, ImagePyramid *hintSurface=NULL);
void UnRegisterSurfaces(NvMOfaFlow *pNvMOfa, ImagePyramid *image1, ImagePyramid *image2, ImagePyramid *outSurface, ImagePyramid *costSurface, ImagePyramid *hintSurface=NULL);
bool RegisterSurfaces(NvMOfaFlow *pNvMOfa, ImagePyramid *outSurface, ImagePyramid *costSurface, ImagePyramid *hintSurface=NULL);
void UnRegisterSurfaces(NvMOfaFlow *pNvMOfa, ImagePyramid *outSurface, ImagePyramid *costSurface, ImagePyramid *hintSurface=NULL);
#endif

