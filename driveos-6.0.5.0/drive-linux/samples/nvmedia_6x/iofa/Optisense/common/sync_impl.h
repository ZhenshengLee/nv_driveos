/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef SYNC_IMPL_H
#define SYNC_IMPL_H

#include "common_defs.h"
#include "nvscisync.h"
#include "nvmedia_iofa.h"

typedef enum
{
    WAITER_CPU = 0,
    WAITER_OFA
} waiterSync;

typedef enum
{
    SIGNALER_CPU = 0,
    SIGNALER_OFA
} signalerSync;

class NvSciSyncAttrListClass
{
public:
    NvSciSyncAttrListClass (NvSciSyncModule sciSyncModule)
        : m_attrList (NULL)
    {
        NvSciError err;
        err = NvSciSyncAttrListCreate(sciSyncModule, &m_attrList);
        if (err != NvSciError_Success)
        {
            cerr << "NvSciSyncAttrListCreate failed " << err <<"\n";

        }
    };

    ~NvSciSyncAttrListClass()
    {
        NvSciSyncAttrListFree(m_attrList);
    };

    NvSciSyncAttrList getSyncAttrList()
    {
        return m_attrList;
    };

protected:
    NvSciSyncAttrList m_attrList;
};

class NvSciSyncAttrListCpuSignaler
    : public NvSciSyncAttrListClass
{
public:
    NvSciSyncAttrListCpuSignaler(NvSciSyncModule sciSyncModule)
        : NvSciSyncAttrListClass(sciSyncModule)
    {
        NvSciSyncAttrKeyValuePair keyValue[2];
    bool cpuAccess = true;

        keyValue[0].attrKey         = NvSciSyncAttrKey_NeedCpuAccess;
        keyValue[0].value           = (void *)&cpuAccess;
        keyValue[0].len             = sizeof(cpuAccess);
        NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_SignalOnly;
        keyValue[1].attrKey         = NvSciSyncAttrKey_RequiredPerm;
        keyValue[1].value           = (void *)&cpuPerm;
        keyValue[1].len             = sizeof(cpuPerm);
        NvSciSyncAttrListSetAttrs(m_attrList, keyValue, 2);
    }
};

class NvSciSyncAttrListCpuWaiter
    : public NvSciSyncAttrListClass
{
public:
    NvSciSyncAttrListCpuWaiter(NvSciSyncModule sciSyncModule)
        : NvSciSyncAttrListClass(sciSyncModule)
    {
        NvSciSyncAttrKeyValuePair keyValue[2];
        bool cpuAccess = true;

        keyValue[0].attrKey         = NvSciSyncAttrKey_NeedCpuAccess;
        keyValue[0].value           = (void *)&cpuAccess;
        keyValue[0].len             = sizeof(cpuAccess);
        NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
        keyValue[1].attrKey         = NvSciSyncAttrKey_RequiredPerm;
        keyValue[1].value           = (void *)&cpuPerm;
        keyValue[1].len             = sizeof(cpuPerm);
        NvSciSyncAttrListSetAttrs(m_attrList, keyValue, 2);
    }
};

class NvSciSyncAttrListOfaWaiter
    : public NvSciSyncAttrListClass
{
public:
    NvSciSyncAttrListOfaWaiter(NvSciSyncModule sciSyncModule)
        : NvSciSyncAttrListClass(sciSyncModule)
        , m_ofa(NULL)
    {
        m_ofa = NvMediaIOFACreate();
        NvMediaIOFAFillNvSciSyncAttrList(m_ofa, m_attrList, NVMEDIA_WAITER);
    }

    ~NvSciSyncAttrListOfaWaiter()
    {
        NvMediaIOFADestroy(m_ofa);
    }

private:
    NvMediaIofa *m_ofa;
};

class NvSciSyncAttrListOfaSignaler
    : public NvSciSyncAttrListClass
{
public:
    NvSciSyncAttrListOfaSignaler(NvSciSyncModule sciSyncModule)
        : NvSciSyncAttrListClass(sciSyncModule)
        , m_ofa(NULL)
    {
        m_ofa = NvMediaIOFACreate();
        NvMediaIOFAFillNvSciSyncAttrList(m_ofa, m_attrList, NVMEDIA_SIGNALER);
    }

    ~NvSciSyncAttrListOfaSignaler()
    {
        NvMediaIOFADestroy(m_ofa);
    }

private:
    NvMediaIofa *m_ofa;
};

class NvSciSyncImpl
{
public:
    NvSciSyncImpl(waiterSync waiter, signalerSync signaler)
        : m_sciSyncModule(NULL)
        , m_syncObj(NULL)
    {
        NvSciError err;

        m_waiter= waiter;
        m_signaler = signaler;
        err = NvSciSyncModuleOpen(&m_sciSyncModule);
        if (err != NvSciError_Success)
        {
            cerr << "NvSciSyncModuleOpen failed \n";
        }
        else if (m_waiter == WAITER_CPU)
        {
            err = NvSciSyncCpuWaitContextAlloc(m_sciSyncModule, &m_cpuWaitContext);
            if (err != NvSciError_Success)
            {
                cerr << "NvSciSyncCpuWaitContextAlloc failed \n";
            }
        }
    }

    ~NvSciSyncImpl()
    {
        if (NULL != m_cpuWaitContext)
        {
            NvSciSyncCpuWaitContextFree(m_cpuWaitContext);
        }
        if (m_sciSyncModule != NULL)
        {
            NvSciSyncModuleClose(m_sciSyncModule);
        }
    }

    NvMediaStatus alloc()
    {
        NvSciSyncAttrList unreconciledList[2] = {NULL};
        NvSciSyncAttrList reconciledList = NULL;
        NvSciSyncAttrList newConflictList = NULL;
        NvSciSyncAttrListClass *waiterAttrList = NULL;
        NvSciSyncAttrListClass *signalerAttrList = NULL;
        NvMediaStatus status = NVMEDIA_STATUS_OK;
        NvSciError err;

        if (m_waiter == WAITER_CPU)
        {
            waiterAttrList = new  NvSciSyncAttrListCpuWaiter(m_sciSyncModule);
        }
        else
        {
            waiterAttrList = new  NvSciSyncAttrListOfaWaiter(m_sciSyncModule);
        }
        if (waiterAttrList == NULL)
        {
           cerr << "waiterAttrList class creation failed \n";
           status = NVMEDIA_STATUS_ERROR;
           goto exit;
        }
        if (m_signaler == SIGNALER_CPU)
        {
            signalerAttrList = new  NvSciSyncAttrListCpuSignaler(m_sciSyncModule);
        }
        else
        {
            signalerAttrList = new  NvSciSyncAttrListOfaSignaler(m_sciSyncModule);
        }
        if (signalerAttrList == NULL)
        {
           cerr << "signalerAttrList class creation failed \n";
           status = NVMEDIA_STATUS_ERROR;
           goto exit;
        }

        unreconciledList[0] = waiterAttrList->getSyncAttrList();
        unreconciledList[1] = signalerAttrList->getSyncAttrList();

        err = NvSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList, &newConflictList);
        if (err != NvSciError_Success)
        {
           cerr << "NvSciSyncAttrListReconcile failed \n";
           status = NVMEDIA_STATUS_ERROR;
           goto exit;
        }

        err = NvSciSyncObjAlloc(reconciledList, &m_syncObj);
        if (err != NvSciError_Success)
        {
           cerr << "NvSciSyncObjAlloc failed \n";
           status = NVMEDIA_STATUS_ERROR;
           goto exit;
        }
        cout << "sync object allocation successful \n";
exit:
        if (reconciledList != NULL)
        {
           NvSciSyncAttrListFree(reconciledList);
        }
        if (newConflictList != NULL)
        {
           NvSciSyncAttrListFree(newConflictList);
        }
        if (waiterAttrList)
        {
            delete waiterAttrList;
        }
        if (signalerAttrList)
        {
            delete signalerAttrList;
        }

        return status;
    }

    NvSciSyncObj getSyncObj()
    {
        return m_syncObj;
    }

    bool checkOpDone(NvSciSyncFence *preFence)
    {
        NvSciError err;

        err = NvSciSyncFenceWait(preFence, m_cpuWaitContext, 100*1000*1000);
        if (err == NvSciError_Success)
        {
            NvSciSyncFenceClear(preFence);
            return true;
        }
        else if (err == NvSciError_Timeout)
        {
            cerr << "opDone failed with timeout error \n";
            return false;
        }
        else
        {
            cerr << "opDone failed with other error \n";
            return false;
        }
    }

    bool generateFence(NvSciSyncFence *preFence)
    {
        NvSciError err;
        if (m_signaler == SIGNALER_CPU)
        {
            err = NvSciSyncObjGenerateFence(m_syncObj, preFence);
            if (err == NvSciError_Success)
            {
                return true;
            }
            else
            {
                cerr << "NvSciSyncObjGenerateFence failed \n";
                return false;
            }
        }
        else
        {
            cerr << "generate fence is not supported\n";
            return false;
	}
    }

    bool signalSyncObj()
    {
        NvSciError err;
        if (m_signaler == SIGNALER_CPU)
        {
            err = NvSciSyncObjSignal(m_syncObj);
            if (err == NvSciError_Success)
            {
                return true;
            }
            else
            {
                cerr << "NvSciSyncObjSignal failed \n";
                return false;
            }
        }
        else
        {
            cerr << "signalSyncObj is not supported\n";
            return false;
	}
    }

    void free()
    {
        if (m_syncObj != NULL)
        {
            NvSciSyncObjFree(m_syncObj);
        }
    }

private:
    NvSciSyncModule         m_sciSyncModule;
    NvSciSyncObj            m_syncObj;
    NvSciSyncCpuWaitContext m_cpuWaitContext;
    waiterSync              m_waiter;
    signalerSync            m_signaler;
};

#endif
