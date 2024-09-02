/* NvSciStream Safety Sample App - ASIL process
 *
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscistream.h"
#include "block_info.h"
#include "usecase.h"
#include "util.h"

/* Create Stream blocks for ASIL process */
static int32_t createStreamBlocks(uint32_t useMailbox)
{
    NvSciError err;

    err = NvSciStreamStaticPoolCreate(testArgs.numPacket, &staticPool);
    if (err != NvSciError_Success) {
        return -1;
    }

    err = NvSciStreamProducerCreate(staticPool, &producer);
    if (err != NvSciError_Success) {
        return -1;
    }

    if (testArgs.numCons > 1U) {
        err =  NvSciStreamMulticastCreate(testArgs.numCons, &multicast);
        if (err != NvSciError_Success) {
            return -1;
        }
    }

    err = NvSciStreamIpcSrcCreate(
            ipcsrcEndpoint,
            sciSyncModule,
            sciBufModule,
            &ipcSrc[0]);
    if (err != NvSciError_Success) {
        return -1;
    }

    err = useMailbox
                   ? NvSciStreamMailboxQueueCreate(&queue[0])
                   : NvSciStreamFifoQueueCreate(&queue[0]);

    if (err != NvSciError_Success) {
        return -1;
    }

    err = NvSciStreamConsumerCreate(queue[0], &consumer);
    if (err != NvSciError_Success) {
        return -1;
    }


    /* Add endpoint information on consumer side.
     *  Application can specify user-defined info to help set up stream,
     *  which can be queried by other blocks after stream connection.
     */
    char info[INFO_SIZE] = {0};
    size_t infoSize =
        snprintf(info, INFO_SIZE, "%s%d", "Consumer proc: ", getpid());
    err = NvSciStreamBlockUserInfoSet(consumer,
                                      ENDINFO_NAME_PROC,
                                      infoSize, info);
    if (err != NvSciError_Success) {
        return -1;
    }

    /* Create a limiter block */
    err = NvSciStreamLimiterCreate(testArgs.limit, &limiter);
    if (err != NvSciError_Success) {
        return -1;
    }

    /* Create a returnSync block */
    err = NvSciStreamReturnSyncCreate(sciSyncModule, &returnSync);
    if (err != NvSciError_Success) {
        return -1;
    }


    /* Add endpoint information on producer side.
     *  Application can specify user-defined info to help set up stream,
     *  which can be queried by other blocks after stream connection.
     */
    infoSize =
        snprintf(info, INFO_SIZE, "%s%d", "Producer proc: ", getpid());
    err = NvSciStreamBlockUserInfoSet(producer,
                                      ENDINFO_NAME_PROC,
                                      infoSize, info);
    if (err != NvSciError_Success) {
        return -1;
    }

    return 0;
}

/* Destory ASIL stream */
static void destroyStream(void)
{

    if (producer != 0U) {
        NvSciStreamBlockDelete(producer);
        producer = 0U;
    }


    if (staticPool != 0U) {
        NvSciStreamBlockDelete(staticPool);
        staticPool = 0U;
    }

    if (testArgs.numCons > 1U) {
        if (multicast != 0U) {
            NvSciStreamBlockDelete(multicast);
            multicast = 0U;
        }
    }

    if (ipcSrc[0] != 0U) {
        NvSciStreamBlockDelete(ipcSrc[0]);
        ipcSrc[0] = 0U;
    }

    if (queue[0] != 0U) {
        NvSciStreamBlockDelete(queue[0]);
        queue[0] = 0U;
    }

    if (consumer != 0U) {
        NvSciStreamBlockDelete(consumer);
        consumer = 0U;
    }

    if (limiter != 0U) {
        NvSciStreamBlockDelete(limiter);
        limiter = 0U;
    }

    if (returnSync != 0U) {
        NvSciStreamBlockDelete(returnSync);
        returnSync = 0U;
    }
}

/* Connect ASIL stream blocks */
static int32_t createStream(void)
{

    NvSciError err;

    err = NvSciStreamBlockConnect(producer, multicast);
    if (err != NvSciError_Success) {
        return -1;
    }

    err = NvSciStreamBlockConnect(multicast, consumer);
    if (err != NvSciError_Success) {
        return -1;
    }

    err = NvSciStreamBlockConnect(multicast, limiter);
    if (err != NvSciError_Success) {
        return -1;
    }

    err = NvSciStreamBlockConnect(limiter, returnSync);
    if (err != NvSciError_Success) {
        return -1;
    }

    err = NvSciStreamBlockConnect(returnSync, ipcSrc[0]);
    if (err != NvSciError_Success) {
        return -1;
    }

    // Cannot predict order of upstream / downstream connects
    // so poll for all of them.
    // Upstream connects are async - so have to loop longer.
    bool consConnected = false;
    bool prodConnected = false;
    bool poolConnected = false;
    bool multicastConnected = false;
    bool queueConnected = false;
    bool limiterConnected = false;
    bool returnsyncConnected = false;
    bool ipcsrcConnected = false;

    NvSciStreamEventType event;

    for (uint32_t i = 0U; i < 50; i++) {
        if (!prodConnected) {
            err = NvSciStreamBlockEventQuery(
                producer,
                testArgs.timeout,
                &event);
            if (err != NvSciError_Timeout) {
                prodConnected =
                    (event == NvSciStreamEventType_Connected);
            }
        }

        if (!consConnected) {
            err = NvSciStreamBlockEventQuery(
                consumer,
                testArgs.timeout,
                &event);
            if (err != NvSciError_Timeout) {
                consConnected =
                    (event == NvSciStreamEventType_Connected);
            }
        }

        if (!poolConnected) {
            err = NvSciStreamBlockEventQuery(staticPool,
                                             testArgs.timeout,
                                             &event);
            if (err != NvSciError_Timeout) {
                poolConnected = (event == NvSciStreamEventType_Connected);
            }
        }

        // multicast
        if (!multicastConnected) {
            err = NvSciStreamBlockEventQuery(multicast,
                                            testArgs.timeout,
                                            &event);
            if (err != NvSciError_Timeout) {
                multicastConnected =
                    (event == NvSciStreamEventType_Connected);
            }
        }

        if (!queueConnected) {
            err = NvSciStreamBlockEventQuery(queue[0], testArgs.timeout, &event);
            if (err != NvSciError_Timeout) {
                queueConnected =
                    (event == NvSciStreamEventType_Connected);
            }
        }

        if (!limiterConnected) {
            err = NvSciStreamBlockEventQuery(limiter, testArgs.timeout, &event);
            if (err != NvSciError_Timeout) {
                limiterConnected =
                    (event == NvSciStreamEventType_Connected);
            }
        }

        if (!returnsyncConnected) {
            err = NvSciStreamBlockEventQuery(returnSync, testArgs.timeout, &event);
            if (err != NvSciError_Timeout) {
                returnsyncConnected =
                    (event == NvSciStreamEventType_Connected);
            }
        }

        if (!ipcsrcConnected) {
            err = NvSciStreamBlockEventQuery(ipcSrc[0], testArgs.timeout, &event);
            if (err != NvSciError_Timeout) {
                ipcsrcConnected =
                    (event == NvSciStreamEventType_Connected);
            }
        }

        if (prodConnected && consConnected && poolConnected &&
            multicastConnected && queueConnected && limiterConnected &&
            returnsyncConnected && ipcsrcConnected) {
            break;
        }
    }

    assert(prodConnected);
    assert(consConnected);
    assert(poolConnected);
    assert(multicastConnected);
    assert(queueConnected);
    assert(limiterConnected);
    assert(returnsyncConnected);
    assert(ipcsrcConnected);

    if (prodConnected && consConnected && poolConnected &&
            multicastConnected && queueConnected && limiterConnected &&
            returnsyncConnected && ipcsrcConnected) {
        return 0;
    }

    return -1;
}

/* Handling ASIL process */
int32_t handleASILProcess (void) {

    int32_t rv = 0;
    NvSciError err;
    pthread_t prodTid;
    pthread_t consTid;
    pthread_t poolTid;
    ProdArgs prodArgs;
    PoolArgs poolArgs;
    ConsArgs consArgs;

    /* Open the named channel */
    err = NvSciIpcOpenEndpoint("nvscistream_0", &ipcsrcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to open channel for Ipcsrc\n",
               err);
        return 0;
    }
    err = NvSciIpcResetEndpointSafe(ipcsrcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to reset IPC endpoint", err);
        return 0;
    }

    rv = createStreamBlocks(false);
    if (rv < 0) {
        destroyStream();
        return 0;
    }

    rv = createStream();
    if (rv < 0) {
        destroyStream();
        return 0;
    }

    prodArgs.block = producer;
    prodArgs.isProxyApp = false;

    rv = pthread_create(&prodTid, NULL, handleProducer, &prodArgs);
    if (rv != 0) {
        printf("Failed to spawn producer thread\n");
        destroyStream();
        return 0;
    }

    poolArgs.isC2cPool = false;
    poolArgs.numPacket = testArgs.numPacket;
    poolArgs.block = staticPool;

    rv = pthread_create(&poolTid, NULL, handlePool, &poolArgs);
    if (rv != 0) {
        printf("Failed to spawn pool thread\n");
        destroyStream();
        return 0;
    }

    consArgs.block = consumer;
    consArgs.isProxyApp = false;

    rv = pthread_create(&consTid, NULL, handleConsumer, &consArgs);
    if (rv != 0) {
        printf("Failed to spawn consumer thread\n");
        destroyStream();
        return 0;
    }

    (void)pthread_join(prodTid, NULL);
    (void)pthread_join(poolTid, NULL);
    (void)pthread_join(consTid, NULL);

    destroyStream();

    return 0;
}
