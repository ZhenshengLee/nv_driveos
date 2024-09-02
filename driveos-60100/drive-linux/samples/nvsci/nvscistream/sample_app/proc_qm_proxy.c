/* NvSciStream Safety Sample App - QM process(proxy)
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

/* Create QM(Proxy) process stream blocks */
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

    err = NvSciStreamIpcDstCreate(
            ipcdstEndpoint,
            sciSyncModule,
            sciBufModule,
            &ipcDst[0]);
    if (err != NvSciError_Success) {
        return -1;
    }

    for (uint32_t i=0; i< 2U; i++) {
        err = useMailbox
                   ? NvSciStreamMailboxQueueCreate(&queue[i])
                   : NvSciStreamFifoQueueCreate(&queue[i]);

        if (err != NvSciError_Success) {
            return -1;
        }
    }

    /* Create a C2C src block */
    err = NvSciStreamIpcSrcCreate2(ipcsrcEndpoint,
                                  sciSyncModule,
                                  sciBufModule,
                                  queue[1],
                                  &c2cSrc);
    if (err != NvSciError_Success) {
        return -1;
    }

    err = NvSciStreamConsumerCreate(queue[0], &consumer);
    if (err != NvSciError_Success) {
        return -1;
    }

    /* Add endpoint information on producer side.
     *  Application can specify user-defined info to help set up stream,
     *  which can be queried by other blocks after stream connection.
     */
    char info[INFO_SIZE] = {0};
    size_t infoSize =
        snprintf(info, INFO_SIZE, "%s%d", "Producer proc: ", getpid());
    err = NvSciStreamBlockUserInfoSet(producer,
                                      ENDINFO_NAME_PROC,
                                      infoSize, info);
    if (err != NvSciError_Success) {
        return -1;
    }

    /* Add endpoint information on producer side.
     *  Application can specify user-defined info to help set up stream,
     *  which can be queried by other blocks after stream connection.
     */
    infoSize =
        snprintf(info, INFO_SIZE, "%s%d", "Consumer proc: ", getpid());
    err = NvSciStreamBlockUserInfoSet(consumer,
                                      ENDINFO_NAME_PROC,
                                      infoSize, info);
    if (err != NvSciError_Success) {
        return -1;
    }

    return 0;
}

/* Connect QM(Proxy) process stream blocks */
static int32_t createStream(void)
{

    NvSciError err;
    err = NvSciStreamBlockConnect(ipcDst[0], consumer);
    if (err != NvSciError_Success) {
        return -1;
    }

    err = NvSciStreamBlockConnect(producer, c2cSrc);
    if (err != NvSciError_Success) {
        return -1;
    }

    // Cannot predict order of upstream / downstream connects
    // so poll for all of them.
    // Upstream connects are async - so have to loop longer.
    bool ipcdstConnected = false;
    bool queueConnected[2]  = {false};
    bool consConnected = false;
    bool prodConnected = false;
    bool poolConnected = false;
    bool c2csrcConnected = false;

    NvSciStreamEventType event;


    for (uint32_t i = 0U; i < 50U; i++) {
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

        if (!queueConnected[0]) {
            err = NvSciStreamBlockEventQuery(queue[0], testArgs.timeout, &event);
            if (err != NvSciError_Timeout) {
                queueConnected[0] =
                    (event == NvSciStreamEventType_Connected);
            }
        }

        if (!queueConnected[1]) {
            err = NvSciStreamBlockEventQuery(queue[1], testArgs.timeout, &event);
            if (err != NvSciError_Timeout) {
                queueConnected[1] =
                    (event == NvSciStreamEventType_Connected);
            }
        }

        if (!c2csrcConnected) {
            err = NvSciStreamBlockEventQuery(c2cSrc, testArgs.timeout, &event);
            if (err != NvSciError_Timeout) {
                c2csrcConnected =
                    (event == NvSciStreamEventType_Connected);
            }
        }

    if (!ipcdstConnected) {
        err = NvSciStreamBlockEventQuery(ipcDst[0], testArgs.timeout, &event);
        if (err != NvSciError_Timeout) {
            ipcdstConnected =
                (event == NvSciStreamEventType_Connected);
        }

    }
    if (prodConnected && consConnected && poolConnected &&
            queueConnected[0] && queueConnected[1] &&
            c2csrcConnected && ipcdstConnected) {
            break;
        }
    }

    assert(prodConnected);
    assert(consConnected);
    assert(poolConnected);
    assert(queueConnected[0]);
    assert(queueConnected[1]);
    assert(c2csrcConnected);
    assert(ipcdstConnected);

    if (prodConnected && consConnected && poolConnected &&
            queueConnected[0] && queueConnected[1] && c2csrcConnected &&
            ipcdstConnected) {
        return 0;
    }
    return -1;
}

/* Destroy QM (Proxy) process stream blocks */
static void destoryStream(void) {

    if (queue[0] != 0) {
        NvSciStreamBlockDelete(queue[0]);
        queue[0] = 0;
    }

    if (queue[1] != 0) {
        NvSciStreamBlockDelete(queue[1]);
        queue[1] = 0;
    }

    if (consumer != 0) {
        NvSciStreamBlockDelete(consumer);
        consumer = 0;
    }

    if (staticPool != 0) {
        NvSciStreamBlockDelete(staticPool);
        staticPool = 0;
    }

    if (producer != 0) {
        NvSciStreamBlockDelete(producer);
        producer = 0;
    }

    if (ipcDst[0] != 0) {
        NvSciStreamBlockDelete(ipcDst[0]);
        ipcDst[0] = 0;
    }

    if (c2cSrc != 0) {
        NvSciStreamBlockDelete(c2cSrc);
        c2cSrc = 0;
    }
}

/* Handling QM Proxy process */
int32_t handleQMProxyProcess (void) {

    int32_t rv;
    NvSciError err;
    pthread_t prodTid;
    pthread_t consTid;
    pthread_t poolTid;
    ProdArgs prodArgs;
    PoolArgs poolArgs;
    ConsArgs consArgs;

    /* Open the named channel */
    err = NvSciIpcOpenEndpoint("nvscistream_1", &ipcdstEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to open channel for Ipcdst\n",
               err);
        return 0;
    }
    err = NvSciIpcResetEndpointSafe(ipcdstEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to reset IPC endpoint", err);
        return 0;
    }

    err = NvSciIpcOpenEndpoint(testArgs.c2csrcChannel, &ipcsrcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to open channel for C2CSrc\n",
               err);
        return 0;
    }
    err = NvSciIpcResetEndpointSafe(ipcsrcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to reset IPC endpoint", err);
        return 0;
    }

    rv = createStreamBlocks(false);
    if (rv < 0U) {
        destoryStream();
        return 0;
    }

    rv = createStream();
    if (rv < 0U) {
        destoryStream();
        return 0;
    }

    prodArgs.block = producer;
    prodArgs.isProxyApp = true;
    rv = pthread_create(&prodTid, NULL, handleProducer, &prodArgs);
    if (rv != 0) {
        destoryStream();
        printf("Failed to spawn producer thread\n");
        return 0;
    }

    consArgs.block = consumer;
    consArgs.isProxyApp = true;
    rv = pthread_create(&consTid, NULL, handleConsumer, &consArgs);
    if (rv != 0) {
        destoryStream();
        printf("Failed to spawn consumer thread\n");
        return 0;
    }

    poolArgs.isC2cPool = false;
    poolArgs.numPacket = testArgs.numPacket;
    poolArgs.block = staticPool;

    rv = pthread_create(&poolTid, NULL, handlePool, &poolArgs);
    if (rv != 0) {
        destoryStream();
        printf("Failed to spawn pool thread\n");
        return 0;
    }

    (void)pthread_join(prodTid, NULL);
    (void)pthread_join(poolTid, NULL);
    (void)pthread_join(consTid, NULL);
    destoryStream();

    return 0;
}
