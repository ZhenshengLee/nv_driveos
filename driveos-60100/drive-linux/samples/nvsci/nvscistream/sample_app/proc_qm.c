/* NvSciStream Safety Sample App - QM process
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

/* Create QM process stream blocks */
static int32_t createStreamBlocks(uint32_t useMailbox)
{

    NvSciError err;
    err = NvSciStreamStaticPoolCreate(testArgs.numPacket, &staticPool);
    if (err != NvSciError_Success) {
        return -1;
    }

    err = NvSciStreamIpcDstCreate2(ipcdstEndpoint,
                                  sciSyncModule,
                                  sciBufModule,
                                  staticPool,
                                  &c2cDst);
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

    return 0;
}

/* Connect QM process stream blocks */
static int32_t createStream(void)
{

    NvSciError err;

    err = NvSciStreamBlockConnect(c2cDst, consumer);
    if (err != NvSciError_Success) {
        return -1;
    }


    // Cannot predict order of upstream / downstream connects
    // so poll for all of them.
    // Upstream connects are async - so have to loop longer.
    bool c2cdstConnected = false;
    bool queueConnected  = false;
    bool consConnected = false;
    bool poolConnected = false;

    NvSciStreamEventType event;

    for (uint32_t i = 0U; i < 50U; i++) {
        if (!c2cdstConnected) {
            err = NvSciStreamBlockEventQuery(
                c2cDst,
                testArgs.timeout,
                &event);
            if (err != NvSciError_Timeout) {
                c2cdstConnected =
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

        if (!queueConnected) {
            err = NvSciStreamBlockEventQuery(queue[0], testArgs.timeout, &event);
            if (err != NvSciError_Timeout) {
                queueConnected =
                    (event == NvSciStreamEventType_Connected);
            }
        }

        if (consConnected && poolConnected &&
            queueConnected && c2cdstConnected) {
            break;
        }
    }

    assert(c2cdstConnected);
    assert(consConnected);
    assert(poolConnected);
    assert(queueConnected);

    if (consConnected && poolConnected &&
            queueConnected && c2cdstConnected) {
        return 0;
    }

    return -1;
}

/* Destroy QM process stream blocks */
static void destoryStream(void) {

    if (staticPool != 0U) {
        NvSciStreamBlockDelete(staticPool);
        staticPool = 0U;
    }

    if (c2cDst != 0U) {
        NvSciStreamBlockDelete(c2cDst);
        c2cDst = 0U;
    }

    if (queue[0] != 0U) {
        NvSciStreamBlockDelete(queue[0]);
        queue[0] = 0U;
    }

    if (consumer != 0U) {
        NvSciStreamBlockDelete(consumer);
        consumer = 0U;
    }
}

/* Handling QM process */
int32_t handleQMProcess (void) {

    int32_t rv;
    NvSciError err;
    pthread_t consTid;
    pthread_t poolTid;
    PoolArgs poolArgs;
    ConsArgs consArgs;

    err = NvSciIpcOpenEndpoint(testArgs.c2cdstChannel, &ipcdstEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to open channel for C2CDst\n",
               err);
        return 0;
    }
    err = NvSciIpcResetEndpointSafe(ipcdstEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to reset IPC endpoint", err);
        return 0;
    }

    rv = createStreamBlocks(true);
    if (rv < 0U) {
        destoryStream();
        return 0;
    }

    rv = createStream();
    if (rv < 0U) {
        destoryStream();
        return 0;
    }

    consArgs.block = consumer;
    consArgs.isProxyApp = false;

    rv = pthread_create(&consTid, NULL, handleConsumer, &consArgs);
    if (rv != 0) {
        destoryStream();
        printf("Failed to spawn consumer thread\n");
        return 0;
    }

    poolArgs.isC2cPool = true;
    poolArgs.numPacket = testArgs.numPacket;
    poolArgs.block = staticPool;

    rv = pthread_create(&poolTid, NULL, handlePool, &poolArgs);
    if (rv != 0) {
        destoryStream();
        printf("Failed to spawn pool thread\n");
        return 0;
    }

    (void)pthread_join(poolTid, NULL);
    (void)pthread_join(consTid, NULL);
    destoryStream();

    return 0;
}
