/* Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifdef NVMEDIA_NVSCI_ENABLE
#include "nvscierror.h"
#include "nvscibuf.h"
#include "nvmedia_core.h"
#include "nvmedia_surface.h"
#include "nvmedia_image.h"
#include "nvmedia_image_nvscibuf.h"
#include "image_scibuf_utils.h"
#include "log_utils.h"

NvMediaImage *
NvMediaImageCreateUsingNvScibuf(
    NvMediaDevice *device,
    NvMediaSurfaceType type,
    const NvMediaSurfAllocAttr *attrs,
    uint32_t numAttrs,
    uint32_t flags
)
{
    NvSciBufModule module = NULL;
    NvSciError err = NvSciError_Success;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvSciBufAttrList attrlist = NULL;
    NvSciBufAttrList conflictlist = NULL;
    NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair attr_kvp = {NvSciBufGeneralAttrKey_RequiredPerm,
                                         &access_perm,
                                         sizeof(access_perm)};
    NvSciBufObj bufobj = NULL;
    NvMediaImage *image = NULL;

    err = NvSciBufModuleOpen(&module);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: NvSciBuffModuleOpen failed. Error: %d \n", __func__, err);
        goto fail_cleanup;
    }

    err = NvSciBufAttrListCreate(module, &attrlist);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: SciBufAttrListCreate failed. Error: %d \n", __func__, err);
        goto fail_cleanup;
    }

    err = NvSciBufAttrListSetAttrs(attrlist, &attr_kvp, 1);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: AccessPermSetAttr failed. Error: %d \n", __func__, err);
        goto fail_cleanup;
    }

    status = NvMediaImageFillNvSciBufAttrs(device,
                                           type,
                                           attrs,
                                           numAttrs,
                                           0,
                                           attrlist);


    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: ImageFillSciBufAttrs failed. Error: %d \n", __func__, status);
        goto fail_cleanup;
    }

    err = NvSciBufAttrListReconcileAndObjAlloc(&attrlist,
                                               1,
                                               &bufobj,
                                               &conflictlist);

    if(err != NvSciError_Success) {
        LOG_ERR("%s: ScibuffAttrlistReconcileAndObjAlloc failed. Error: %d \n", __func__, err);
        NvSciBufAttrListFree(conflictlist);
        goto fail_cleanup;
    }

    status = NvMediaImageCreateFromNvSciBuf(device,
                                            bufobj,
                                            &image);

    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: ImageCreatefromSciBuf failed. Error: %d \n", __func__, status);
        goto fail_cleanup;
    }

    NvSciBufAttrListFree(attrlist);

    if(bufobj != NULL) {
        NvSciBufObjFree(bufobj);
    }

    if(module != NULL) {
        NvSciBufModuleClose(module);
    }

    return image;

fail_cleanup:
    if(attrlist != NULL) {
        NvSciBufAttrListFree(attrlist);
    }
    if(bufobj != NULL) {
        NvSciBufObjFree(bufobj);
        bufobj = NULL;
    }

    if(module != NULL) {
        NvSciBufModuleClose(module);
    }
    NvMediaImageDestroy(image);
    return NULL;
}
#endif
