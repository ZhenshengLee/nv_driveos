/* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "cmn_functions.h"

bool RegisterSurfaces(NvMOfa *pNvMOfa, Buffer *image1, Buffer *image2, Buffer *outSurface,Buffer *costSurface)
{
    if (!pNvMOfa->registerSurfaceToNvMOfa(image1->getHandle()))
    {
        cerr << "NvMOfa registerSurfaceToNvMOfa api for image1 failed\n";
        return false;
    }
    if (!pNvMOfa->registerSurfaceToNvMOfa(image2->getHandle()))
    {
        cerr << "NvMOfa registerSurfaceToNvMOfa api for image2 failed\n";
        return false;
    }
    if (!pNvMOfa->registerSurfaceToNvMOfa(outSurface->getHandle()))
    {
        cerr << "NvMOfa registerSurfaceToNvMOfa api for output failed\n";
        return false;
    }
    if (!pNvMOfa->registerSurfaceToNvMOfa(costSurface->getHandle()))
    {
        cerr << "NvMOfa registerSurfaceToNvMOfa api for cost failed\n";
        return false;
    }
    return true;
}

bool RegisterSurfaces(NvMOfa *pNvMOfa, Buffer *outSurface,Buffer *costSurface)
{
    if (!pNvMOfa->registerSurfaceToNvMOfa(outSurface->getHandle()))
    {
        cerr << "NvMOfa registerSurfaceToNvMOfa api for output failed\n";
        return false;
    }
    if (!pNvMOfa->registerSurfaceToNvMOfa(costSurface->getHandle()))
    {
        cerr << "NvMOfa registerSurfaceToNvMOfa api for cost failed\n";
        return false;
    }
    return true;
}

void UnRegisterSurfaces(NvMOfa *pNvMOfa, Buffer *image1, Buffer *image2, Buffer *outSurface, Buffer *costSurface)
{
    if (image1 != NULL)
    {
        if (!pNvMOfa->unRegisterSurfaceToNvMOfa(image1->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api for image1 failed\n";
        }
    }
    if (image2 != NULL)
    {
        if (!pNvMOfa->unRegisterSurfaceToNvMOfa(image2->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api for image2 failed\n";
        }
    }

    if (outSurface != NULL)
    {
        if (!pNvMOfa->unRegisterSurfaceToNvMOfa(outSurface->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api failed\n";
        }
    }

    if (costSurface != NULL)
    {
        if (!pNvMOfa->unRegisterSurfaceToNvMOfa(costSurface->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api failed\n";
        }
    }
}

void UnRegisterSurfaces(NvMOfa *pNvMOfa, Buffer *outSurface, Buffer *costSurface)
{
    if (outSurface != NULL)
    {
        if (!pNvMOfa->unRegisterSurfaceToNvMOfa(outSurface->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api failed\n";
        }
    }

    if (costSurface != NULL)
    {
        if (!pNvMOfa->unRegisterSurfaceToNvMOfa(costSurface->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api failed\n";
        }
    }
}

bool writeBinFile (string filename, char *surface, uint32_t size)
{
    fstream pfile;
    bool status = false;

    pfile.open(filename, ios::out | ios::binary);
    if (pfile.is_open())
    {
        if (!pfile.write(surface, size))
        {
            cerr << "writeBinFile: " << filename << " failed\n";
            status = false;
        }
        else
        {
            cout << "writeBinFile: " << filename << " done\n";
            status = true;
        }
        pfile.close();
    }

    return status;
}


bool RegisterSurfaces (NvMOfaFlow *pNvMOfa, ImagePyramid *image1, ImagePyramid *image2, ImagePyramid *outSurface, ImagePyramid *costSurface, ImagePyramid *hintSurface)
{
    uint16_t i;
    for (i = 0; i < pNvMOfa->getNumOfPydLevels(); i++)
    {
        if (!pNvMOfa->registerSurfaceToNvMOfa((image1->getImageBuffer(i))->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api for image1 failed for level " << i << "\n";
            return false;
        }
        if (!pNvMOfa->registerSurfaceToNvMOfa((image2->getImageBuffer(i))->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api for image2 failed for level " << i << "\n";
            return false;
        }

        if (!pNvMOfa->registerSurfaceToNvMOfa((outSurface->getImageBuffer(i))->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api for output failed for level " << i << "\n";
            return false;
        }
        if (!pNvMOfa->registerSurfaceToNvMOfa((costSurface->getImageBuffer(i))->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api for cost failed for level " << i << "\n";
            return false;
        }
    }

    if (hintSurface != NULL)
    {
        for (i = 0; i < pNvMOfa->getNumOfPydLevels()-1; i++)
        {
            if (!pNvMOfa->registerSurfaceToNvMOfa((hintSurface->getImageBuffer(i))->getHandle()))
            {
                cerr << "NvMOfa registerSurfaceToNvMOfa api for cost failed for level " << i << "\n";
                return false;
            }
        }
    }

    return true;
}

void UnRegisterSurfaces (NvMOfaFlow *pNvMOfa, ImagePyramid *image1, ImagePyramid *image2, ImagePyramid *outSurface, ImagePyramid *costSurface, ImagePyramid *hintSurface)
{
    uint16_t i;

    for (i = 0; i < pNvMOfa->getNumOfPydLevels(); i++)
    {
        if (image1 != NULL)
        {
            if (!pNvMOfa->unRegisterSurfaceToNvMOfa((image1->getImageBuffer(i))->getHandle()))
            {
                cerr << "NvMOfa registerSurfaceToNvMOfa api for image1 failed\n";
            }
        }
        if (image2 != NULL)
        {
            if (!pNvMOfa->unRegisterSurfaceToNvMOfa((image2->getImageBuffer(i))->getHandle()))
            {
                cerr << "NvMOfa registerSurfaceToNvMOfa api for image2 failed\n";
            }
        }
        if (outSurface != NULL)
        {
            if (!pNvMOfa->unRegisterSurfaceToNvMOfa((outSurface->getImageBuffer(i))->getHandle()))
            {
                cerr << "NvMOfa registerSurfaceToNvMOfa api failed\n";
            }
        }
        if (costSurface != NULL)
        {
            if (!pNvMOfa->unRegisterSurfaceToNvMOfa((costSurface->getImageBuffer(i))->getHandle()))
            {
                cerr << "NvMOfa registerSurfaceToNvMOfa api failed\n";
            }
        }
    }
    for (i = 0; i < pNvMOfa->getNumOfPydLevels()-1; i++)
    {
        if (hintSurface != NULL)
        {
            if (!pNvMOfa->unRegisterSurfaceToNvMOfa((hintSurface->getImageBuffer(i))->getHandle()))
            {
                cerr << "NvMOfa registerSurfaceToNvMOfa api failed\n";
            }
        }
    }
}

bool RegisterSurfaces (NvMOfaFlow *pNvMOfa, ImagePyramid *outSurface, ImagePyramid *costSurface, ImagePyramid *hintSurface)
{
    uint16_t i;
    for (i = 0; i < pNvMOfa->getNumOfPydLevels(); i++)
    {
        if (!pNvMOfa->registerSurfaceToNvMOfa((outSurface->getImageBuffer(i))->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api for output failed for level " << i << "\n";
            return false;
        }
        if (!pNvMOfa->registerSurfaceToNvMOfa((costSurface->getImageBuffer(i))->getHandle()))
        {
            cerr << "NvMOfa registerSurfaceToNvMOfa api for cost failed for level " << i << "\n";
            return false;
        }
    }

    if (hintSurface != NULL)
    {
        for (i = 0; i < pNvMOfa->getNumOfPydLevels()-1; i++)
        {
            if (!pNvMOfa->registerSurfaceToNvMOfa((hintSurface->getImageBuffer(i))->getHandle()))
            {
                cerr << "NvMOfa registerSurfaceToNvMOfa api for cost failed for level " << i << "\n";
                return false;
            }
        }
    }
 return true;
}

void UnRegisterSurfaces (NvMOfaFlow *pNvMOfa, ImagePyramid *outSurface, ImagePyramid *costSurface, ImagePyramid *hintSurface)
{
    uint16_t i;

    for (i = 0; i < pNvMOfa->getNumOfPydLevels(); i++)
    {
        if (outSurface != NULL)
        {
            if (!pNvMOfa->unRegisterSurfaceToNvMOfa((outSurface->getImageBuffer(i))->getHandle()))
            {
                cerr << "NvMOfa registerSurfaceToNvMOfa api failed\n";
            }
        }
        if (costSurface != NULL)
        {
            if (!pNvMOfa->unRegisterSurfaceToNvMOfa((costSurface->getImageBuffer(i))->getHandle()))
            {
                cerr << "NvMOfa registerSurfaceToNvMOfa api failed\n";
            }
        }
    }
    for (i = 0; i < pNvMOfa->getNumOfPydLevels()-1; i++)
    {
        if (hintSurface != NULL)
        {
            if (!pNvMOfa->unRegisterSurfaceToNvMOfa((hintSurface->getImageBuffer(i))->getHandle()))
            {
                cerr << "NvMOfa registerSurfaceToNvMOfa api failed\n";
            }
        }
    }
}

