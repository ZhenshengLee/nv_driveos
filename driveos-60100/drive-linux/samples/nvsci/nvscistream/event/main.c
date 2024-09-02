/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/*
 * NvSciStream Event Loop Driven Sample App - main application
 *
 * Application info:
 *  This application creates a single stream with up to 4 consumers.
 *  Depending on command line options, the consumers can run in the same
 *    process as the producer, separate processes, or some combination
 *    thereof.
 *  Each consumer can be separately set to use a FIFO or mailbox queue,
 *    and an optional limiter block can be requested for each one.
 *
 *  An application can check for NvSciStream events either by polling
 *    or waiting for a single block at a time through a NvSciStream function,
 *    or by binding an NvSciEventService to the blocks, allowing it to
 *    wait for events on multiple NvSciStream blocks (as well as other
 *    components capable of signalling NvSciEvents) simultaneously.
 *  If an application carefully controls the order of operations, it
 *    may be able to wait for specific events in a single thread. But
 *    more generally, NvSciStream supports an event loop driven model,
 *    where any event may occur and the application responds appropriately.
 *    This can be done either with a single thread handling all blocks,
 *    or separate threads for each block. This application provides
 *    examples of both use cases.
 *
 *  For testing and demonstration purposes, the default target setup for
 *    the NVIDIA SDK includes indexed NvSciIpc channels with base name
 *    "nvscistream_", which are connected together in even/odd pairs.
 *    (So nvscistream_0 is connected to nvscistream_1, nvscistream_2 is
 *    connected to nvscistream_3, and so on.) This sample application is
 *    hard-coded to use these channels when streaming between processes.
 *    A production application should be modified to use the channels
 *    defined for the production target.
 *
 *  This application is intended to illustrate how to do full setup of a
 *    stream, assuming everything is working correctly. It does all
 *    necessary error checking to confirm setup succeeded, but does not
 *    attempt any recovery or do a full teardown in the event a failure
 *    is detected.
 *
 *  Our approach to abstracting the event and per-block support is object
 *    oriented and would lend itself well to C++. But for simplicity,
 *    since NvSciStream itself is a C interface, we have restricted this
 *    sample application to C code.
 *
 *  Unless otherwise stated, all functions in all files return 1 to
 *    indicate success and 0 to indicate failure.
 */

#include <unistd.h>
#include <stdio.h>
#if (QNX == 1)
#include <sys/neutrino.h>
#endif
#include "nvscisync.h"
#include "nvscibuf.h"
#include "nvsciipc.h"
#include "nvscistream.h"
#include "event_loop.h"
#include "block_info.h"
#include <pthread.h>
#include <stdatomic.h>

/* Base name for all IPC channels */
static const char ipcBaseName[] = "nvscistream_";

/* Event handling function table */
EventFuncs const* eventFuncs = NULL;

/* Top level use-case setup function pointers */
int32_t (*createProducer)(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool,
    uint32_t          numFrames) = createProducer_Usecase1;
int32_t (*createConsumer)(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  queue,
    uint32_t          index,
    uint32_t          frames) = createConsumer_Usecase1;

int32_t(*createPool)(
    NvSciStreamBlock* pool,
    uint32_t          numPacket,
    bool              isC2cPool) = createPool_Common;


/* NvSci modules */
NvSciSyncModule sciSyncModule;
NvSciBufModule  sciBufModule;

/* Flag used to terminate the thread that
* was spawned to handle the late/re-attached
* consumer connections upon stream disconnect
*/
atomic_int streamDone;

/* Holds the multicast block handle for
* late/re-attach usecase
*/
static NvSciStreamBlock multicastBlock = 0U;

/* Dispatch thread for handling late/re-attach
*  consumer connections
*/
pthread_t        dispatchThread;

/* Endpoint status structure*/
Endpoint ipcEP[MAX_CONSUMERS];

/* pthread variables for thread synchronization */
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

#if (QNX == 1)
/* Data needed for QNX channel connection */
int8_t DELTA = 1;
int8_t ipcCode = (_PULSE_CODE_MINAVAIL + 1);
#endif
/* Common options for all blocks */
CommonOptions opts;

/* Options for producer */
typedef struct {
    uint32_t resident;
    uint32_t numConsumer;
    uint32_t numPacket;
    uint32_t numFrames;
    uint32_t usecase;
} ProducerOptions;

/* Options for each consumer */
typedef struct {
    uint32_t resident;
    uint32_t useMailbox;
    uint32_t useLimiter;
    uint32_t c2cMode;
    uint32_t c2cSrcUseMailbox;
    uint32_t c2cDstNumPacket;
    char srcChannel[32];
    char dstChannel[32];
    char ipcChannelForHandshake[32];
    uint32_t numFrames;
} ConsumerOptions;

ConsumerOptions consOpts[MAX_CONSUMERS];

/* Print command line options */
static void print_usage(const char *str)
{
    printf("%s [options]\n", str);
    printf(" For single- or inter-process/chip operation:\n");
    printf("   -m <count> [default 1, max %d]\n", MAX_CONSUMERS);
    printf("       number of multicast consumers\n");
    printf("       (ignored if process doesn't own producer\n");
    printf("   -f <count> [default 3]\n");
    printf("       number of packets in main pool\n");
    printf("       (ignored if process doesn't own producer\n");
    printf("   -l <index> <limit> [default - not used]\n");
    printf("       use limiter block for indexed consumer\n");
    printf("       (ignored if process doesn't own producer\n");
    printf("   -q <index> {f|m} [default f]\n");
    printf("       use fifo (f) or maibox (m) for indexed consumer\n");
    printf("       (ignored if process doesn't own the indexed consumer\n");
    printf("   -e {s|t} [default s]\n");
    printf("       s : events are handled through a single service\n");
    printf("       t : events are handled with separate per-block threads\n");
    printf("   -E\n");
    printf("       Use the user-provided event service to handle internal I/O "
                  "messages on ipc blocks.\n");
    printf("       Only Supported with event service\n");
#if (NV_SUPPORT_NVMEDIA == 1)
    printf("   -s {y|r} [default r]\n");
    printf("       y : NvSciColor_Y8U8Y8V8 Image Color Format in use case 2\n");
    printf("       r : NvSciColor_A8R8G8B8 Image Color Format in use case 2\n");
#endif
    printf("   -u <index> [default 1]\n");
    printf("       use case (must be same for all processes)\n");
    printf("       1 : CUDA (rt) producer to CUDA (rt) consumer\n");
#if (NV_SUPPORT_NVMEDIA == 1)
    printf("       2 : NvMedia producer to CUDA (rt) consumer\n");
#endif
#ifndef NV_SUPPORT_DESKTOP
    printf("       3 : CUDA (rt) producer to CUDA (rt) consumer "
                       "in ASIL-D process, not supported in C2C.\n");
#endif
    printf("   -i [default - not used]\n");
    printf("       set endpoint info and query info from other endpoints\n");
    printf(" For inter-process operation:\n");
    printf("   -p\n");
    printf("       producer resides in this process\n");
    printf("   -c <index> \n");
    printf("       indexed consumer resides in this process\n");
    printf(" For inter-chip (C2C) operation:\n");
    printf("   -P <index> <Ipc endpoint name>\n");
    printf("       producer resides in this process\n");
    printf("       Ipc endpoint used by the producer to communicate with the "
                  "indexed chip-to-chip (C2C) consumer\n");
    printf("       User must provide all the C2C endpoints required to "
                  "communicate with the total number of consumers for C2C "
                  "usecase when late-/reattach is chosen.\n");
    printf("   -C <index> <Ipc endpoint name>\n");
    printf("       indexed consumer resides in this process\n");
    printf("       Ipc endpoint used by this chip-to-chip (C2C) consumer\n");
    printf("       -C and -c can't be used simultaneously.\n");
    printf("       (ignored if process owns producer)\n");
    printf("   -F <index> <count> [default 3]\n");
    printf("       number of packets in pool attached to the IpcDst block "
                  "of the indexed C2C consumer\n");
    printf("       set along with the indexed C2C consumer.\n");
    printf("       (ignored if process doesn't own indexed C2C consumer)\n");
    printf("   -Q <index> {f|m} [default f]\n");
    printf("       use fifo (f) or maibox (m) for C2C IpcSrc of indexed "
                   "consumer.\n");
    printf("       Can't specify same index as -c)\n");
    printf("       set in the producer process.\n");
    printf("       (ignored if process doesn't own producer)\n");
    printf("   -r <index> [default 0]\n");
    printf("       Number of late-attach consumers\n");
    printf("       set in the producer process and currently supported "
                   "for usecase1.\n");
    printf("   -L\n");
    printf("       set in the consumer process to indicate the consumer "
                   "connection is late/re-attach.\n");
    printf("   -k <index> <frames> [default 0]\n");
    printf("       Number of frames expected to be received by the indexed consumer\n");
    printf("   -n <frames> [default 32]\n");
    printf("       Number of frames expected to be produced by the producer\n");
    printf("       With -r option is specified, the default value is set to 100000\n");
    printf("       With -r option is NOT specified, the default value is set to 32\n");
}

/* Deletes the block that are created for handling late-/re-attach
* connection when late/re-attach consumer connection fails.
*/
static void deleteBlock(NvSciStreamBlock block)
{
    for (int32_t i=0; i< numBlocks; i++) {
        BlockEventData* entry = &blocks[i];
        if (entry->handle == block) {
            deleteCommon(entry->data);
        }
    }
}

/* Function to handle the opening of IPC endpoint
* for handshaking.
*/
static bool openIpcEndpoint(uint32_t index, bool isConsumer)
{
    NvSciError err;
    if (!isConsumer) {
        err = NvSciIpcOpenEndpoint(ipcEP[index].ipcChannelForHandshake,
                                   &ipcEP[index].ipcEndpoint);
        if (NvSciError_Success != err) {
            printf("Failed (%x) to open channel (%s) for IpcSrc\n",
                err, ipcEP[index].ipcChannelForHandshake);
            return false;
        }
    } else {
        err = NvSciIpcOpenEndpoint(consOpts[index].ipcChannelForHandshake,
                                   &ipcEP[index].ipcEndpoint);
        if (NvSciError_Success != err) {
            printf("Failed (%x) to open channel (%s) for IpcDst\n",
                   err, consOpts[index].ipcChannelForHandshake);
            return false;
        }
    }

#if (QNX == 1)
    ipcEP[index].chid = ChannelCreate_r(_NTO_CHF_UNBLOCK | _NTO_CHF_PRIVATE);
    if (ipcEP[index].chid < 0) {
        printf("ChannelCreate_r: fail for connection index=%d \n", index);
        return false;
    }

    ipcEP[index].coid = ConnectAttach_r(0, 0, ipcEP[index].chid,
                                       _NTO_SIDE_CHANNEL, _NTO_COF_CLOEXEC);
    if (ipcEP[index].coid < 0) {
        printf("ConnectAttach_r: fail for connection index=%d\n",  index);
        return false;
    }
    err = NvSciIpcSetQnxPulseParamSafe(ipcEP[index].ipcEndpoint, ipcEP[index].coid,
                                        SIGEV_PULSE_PRIO_INHERIT,
                                        ipcCode);
    if (err != NvSciError_Success) {
        printf("NvSciIpcSetQnxPulseParamSafe(%x) failed for connection index=%d\n",
                err, index);
        return false;
    }
#endif

    err = NvSciIpcResetEndpointSafe(ipcEP[index].ipcEndpoint);
    if (err != NvSciError_Success) {
        printf("NvSciIpcResetEndpointSafe(%x) failed for connection index=%d\n",
               err, index);
        return false;
    }

    return true;
}

/* Function to handle the IPC connection for handshaking */
static bool waitForIpcConnection(uint32_t index, bool isConsumer)
{
    NvSciError err;
    bool retry = true;

    while(retry) {
        uint32_t receivedEvents = 0U;
        err = NvSciIpcGetEventSafe(ipcEP[index].ipcEndpoint, &receivedEvents);
        if (NvSciError_Success != err) {
            atomic_store(&streamDone, 1);
            printf("Failed (%x) to retrieve IPC events for connection index=%d\n",
                    err, index);
            return false;
        }
        /* No need to retry if it is a producer */
        if(!isConsumer) {
            retry = false;
        }

        if (receivedEvents & (NV_SCI_IPC_EVENT_CONN_EST_ALL)) {
#if (QNX == 1)
            if (ipcEP[index].coid != 0) {
                (void)ConnectDetach_r(ipcEP[index].coid);
                ipcEP[index].coid = 0;
            }
            if (ipcEP[index].chid != 0) {
                (void)ChannelDestroy_r(ipcEP[index].chid);
                ipcEP[index].chid = 0;
            }
#endif
            err = NvSciIpcCloseEndpointSafe(ipcEP[index].ipcEndpoint, false);
            if (NvSciError_Success != err) {
                atomic_store(&streamDone, 1);
                printf("Failed (%x) to close IPC endpoint for connection index=%d\n",
                        err, index);
                return false;
            }
            ipcEP[index].ipcEndpoint = 0U;

            /* We need to open the endpoint again if it is not a consumer */
            if (!isConsumer) {
                err = NvSciIpcOpenEndpoint(ipcEP[index].ipcChannel,
                                           &ipcEP[index].ipcEndpoint);
                if (NvSciError_Success != err) {
                    atomic_store(&streamDone, 1);
                    printf("Failed (%x) to open channel (%s) for IpcSrc \
                           for connection index=%d\n",
                           err, ipcEP[index].ipcChannel, index);
                    return false;
                }
                err = NvSciIpcResetEndpointSafe(ipcEP[index].ipcEndpoint);
                if (NvSciError_Success != err) {
                    atomic_store(&streamDone, 1);
                    printf("Failed (%x) to reset IPC endpoint for connection \
                            index = %d\n", err, index);
                    return false;
                }
                ipcEP[index].ipcConnected = true;
            }
            return true;
        }
    }
    return false;
}

/* Dispatch thread function to handle late/re-attach
* consumer connections
*/
void* handleLateConsumerThreadFunc(void *args)
{
    NvSciError err;
    bool lateConsumerConnectionFound = false;
    bool retry = false;

    while(!atomic_load(&streamDone) || retry) {
        /* Poll for the status of IPC channels */
        pthread_mutex_lock(&mutex);
        for (uint32_t i=0; i<MAX_CONSUMERS; i++) {
            if (!ipcEP[i].ipcOpened) {
                /* Open the endpoint, which may be using by consumer for late
                * connection at this point, consumer may/may not be waiting for
                * connection
                */
                ipcEP[i].ipcOpened = true;
                if (true != openIpcEndpoint(i, false)) {
                    atomic_store(&streamDone, 1);
                    return NULL;
                }
            }
            if (!ipcEP[i].c2cOpened && opts.c2cMode && consOpts[i].c2cMode &&
                    (i < opts.numConsumer)) {
                /* Open the endpoint, which may be using by consumer for late
                * connection at this point, consumer may/may not be waiting for
                * connection
                */
                /* Open the named channel */
                ipcEP[i].c2cOpened = true;
                err = NvSciIpcOpenEndpoint(ipcEP[i].c2cChannel,
                                           &ipcEP[i].c2cEndpoint);
                if (NvSciError_Success != err) {
                    printf("Failed (%x) to open channel (%s) for c2cSrc\n",
                           err, ipcEP[i].c2cChannel);
                    ipcEP[i].c2cOpened = false;
                }

                err = NvSciIpcResetEndpointSafe(ipcEP[i].c2cEndpoint);
                if (NvSciError_Success != err) {
                    printf("Failed (%x) to reset C2C endpoint", err);
                    ipcEP[i].c2cOpened = false;
                }
            }

            if (!ipcEP[i].ipcConnected) {
                if (true == waitForIpcConnection(i, false)) {
                    /* Create IpcSrc block */
                    if (!createIpcSrc2(&ipcEP[i].ipcBlock,
                                      ipcEP[i].ipcEndpoint,
                                      opts.useExtEventService)) {
                        printf("Failed to create IpcSrc to handle \
                                late consumer connection\n");
                        atomic_store(&streamDone, 1);
                        return NULL;
                    }
                    ++numAlive;
                     /* When Late-/reattach usecase is selected, a returnSync block
                      * is connected to the consumer chain to ensure proper fence waits
                      * during consumer disconnect and reconnect.
                      */
                    if (!createReturnSync(&ipcEP[i].returnSync)) {
                        printf("Failed to create Returnsync to handle late \
                            consumer connection\n");
                        deleteBlock(ipcEP[i].ipcBlock);
                        atomic_store(&streamDone, 1);
                        return NULL;
                    }
                    ++numAlive;
                    /* Connect ReturnSync and IpcSrc */
                    if (NvSciError_Success !=
                        NvSciStreamBlockConnect(ipcEP[i].returnSync,
                                                ipcEP[i].ipcBlock)) {
                        printf("Failed to connect ipcsrc to returnsync\n");
                        deleteBlock(ipcEP[i].ipcBlock);
                        deleteBlock(ipcEP[i].returnSync);
                        atomic_store(&streamDone, 1);
                        return NULL;
                    }
                    /* Connect to multicast */
                    if (NvSciError_Success !=
                        NvSciStreamBlockConnect(multicastBlock,
                                                ipcEP[i].returnSync)) {
                        printf("Failed to connect ipcsrc to multicast\n");
                        deleteBlock(ipcEP[i].ipcBlock);
                        deleteBlock(ipcEP[i].returnSync);
                        atomic_store(&streamDone, 1);
                        return NULL;
                    } else {
                        lateConsumerConnectionFound = true;
                        break;
                    }
                }
            }
            if (!ipcEP[i].c2cConnected && opts.c2cMode && consOpts[i].c2cMode &&
                (i < (opts.numConsumer)))  {
                // Check IPC for connection
                uint32_t receivedEvents = 0U;
                err = NvSciIpcGetEventSafe(ipcEP[i].c2cEndpoint, &receivedEvents);
                if (NvSciError_Success != err) {
                    printf("Failed (%x) to retrieve IPC events", err);
                    atomic_store(&streamDone, 1);
                    return NULL;
                }
                if (receivedEvents & (NV_SCI_IPC_EVENT_CONN_EST_ALL)) {
                    ipcEP[i].c2cConnected = true;

                    if (!createQueue(&ipcEP[i].queue, false)) {
                        printf("Failed to create Queue to handle \
                            late consumer connection\n");
                        atomic_store(&streamDone, 1);
                        return NULL;
                    }
                    ++numAlive;
                    /* Create C2C block */
                    if (!createC2cSrc2(&ipcEP[i].ipcBlock,
                                       ipcEP[i].c2cEndpoint,
                                       ipcEP[i].queue)) {
                        printf("Failed to create C2cSrc to handle late \
                            consumer connection\n");
                        deleteBlock(ipcEP[i].queue);
                        atomic_store(&streamDone, 1);
                        return NULL;
                    }
                    ++numAlive;

                    /* When Late-/reattach usecase is selected, a returnSync
                    * block is connected to the consumer chain to ensure proper
                    * fence waits during consumer disconnect and reconnect.
                    */
                    if (!createReturnSync(&ipcEP[i].returnSync)) {
                        printf("Failed to create Returnsync to handle late \
                            consumer connection\n");
                        deleteBlock(ipcEP[i].ipcBlock);
                        deleteBlock(ipcEP[i].queue);
                        atomic_store(&streamDone, 1);
                        return NULL;
                    }
                    ++numAlive;
                    /* Connect ReturnSync and IpcSrc */
                    if (NvSciError_Success !=
                        NvSciStreamBlockConnect(ipcEP[i].returnSync,
                                                ipcEP[i].ipcBlock)) {
                        printf("Failed to connect c2csrc to returnsync\n");
                        deleteBlock(ipcEP[i].ipcBlock);
                        deleteBlock(ipcEP[i].returnSync);
                        deleteBlock(ipcEP[i].queue);
                        atomic_store(&streamDone, 1);
                        return NULL;
                    }

                    /* Connect to multicast */
                    if (NvSciError_Success !=
                        NvSciStreamBlockConnect(multicastBlock,
                                                ipcEP[i].returnSync)) {
                        printf("Failed to connect returnsync to multicast\n");
                        deleteBlock(ipcEP[i].ipcBlock);
                        deleteBlock(ipcEP[i].returnSync);
                        deleteBlock(ipcEP[i].queue);
                        atomic_store(&streamDone, 1);
                        return NULL;
                    } else {
                        lateConsumerConnectionFound = true;
                        break;
                    }
                }
            }
        }

        if (lateConsumerConnectionFound) {
            /* initiate multicast block to handle the late consumer connection */
            err = NvSciStreamBlockSetupStatusSet(multicastBlock,
                                                NvSciStreamSetup_Connect,
                                                true);
            /* Ignore NvSciError_AlreadyDone and NvSciError_NotYetAvailable
            * error codes as the multicast block might have already started
            * the setup with the connected consumer when it is a last consumer.
            * In case of other consumer connections, the error will be thrown
            * while connecting the new consumer to multicast block during
            * NvSciStreamBlockConnect() call if multicast is not ready yet to
            * accept late/re-attach consumer connections.
            */
            if (err != NvSciError_Success &&
                err != NvSciError_AlreadyDone &&
                err != NvSciError_NotYetAvailable) {
                printf("Attaching a late consumer connection failed = %x\n", err);
                atomic_store(&streamDone, 1);
                return NULL;
            } else {
                lateConsumerConnectionFound = false;
                /* Go for sleep mode until a signal comes from multicast block
                * after processing the current set of late consumers
                */
                pthread_cond_wait(&cond, &mutex);
            }
        }
        pthread_mutex_unlock(&mutex);
        usleep(100000);
    }
    return NULL;
}


/* Set up chain of producer side blocks, up to optional multicast */
static int32_t setupProducerChain(
    NvSciStreamBlock* producerLink,
    ProducerOptions*  prodOpts)
{
    /* Create pool */
    NvSciStreamBlock poolBlock;
    if (!createPool(&poolBlock, prodOpts->numPacket, false)) {
        return 0;
    }

    /* Create producer */
    NvSciStreamBlock producerBlock;
    if (!createProducer(&producerBlock, poolBlock, prodOpts->numFrames)) {
        return 0;
    }

    /* If multicast required, add the block. */
    if (prodOpts->numConsumer > 1) {

        /* Create multicast block */
        if (!createMulticast(&multicastBlock, prodOpts->numConsumer)) {
            return 0;
        }

        /* Connect to producer */
        if (NvSciError_Success !=
            NvSciStreamBlockConnect(producerBlock, multicastBlock)) {
            printf("Failed to connect multicast to producer\n");
            return 0;
        }

        /* Multicast block is end of chain */
        *producerLink = multicastBlock;

    } else {

        /* Producer block is end of chain */
        *producerLink = producerBlock;

    }

    return 1;
}

/* Set up chain of consumer side blocks */
static int32_t setupConsumerChain(
    NvSciStreamBlock* consumerLink,
    ConsumerOptions*  consOpts,
    uint32_t          index)
{
    /*
     * Note: Currently the consumer "chain" just consists of the consumer
     *       itself and its associated queue. We follow this paradigm to
     *       allow easy addition of new optional blocks in the future.
     */

    /* Create queue */
    NvSciStreamBlock queueBlock;
    if (!createQueue(&queueBlock, consOpts->useMailbox)) {
        return 0;
    }

    /* Create consumer */
    NvSciStreamBlock consumerBlock;
    if (!createConsumer(&consumerBlock, queueBlock, index, consOpts->numFrames)) {
        return 0;
    }

    /* Consumer block is start of chain */
    *consumerLink = consumerBlock;

    return 1;
}

/* Add additional branch options */
static int32_t setupBranchOptions(
    NvSciStreamBlock* consumerLink,
    ConsumerOptions*  consOpts)
{
    /* If limiter requested, add it */
    if (consOpts->useLimiter) {

       /* If a consumer may generate unreliable fences, a ReturnSync block can
        * be added as the downstream of the Limiter block for that consumer,
        * to isolate any packets with bad fences.
        */
        NvSciStreamBlock returnSyncBlock;
        if (!createReturnSync(&returnSyncBlock)) {
            return 0;
        }

        /* Connect to incoming consumer chain */
        if (NvSciError_Success !=
            NvSciStreamBlockConnect(returnSyncBlock, *consumerLink)) {
            printf("Failed to connect returnSyncBlock to consumer chain\n");
            return 0;
        }


        /* ReturnSync is new end of chain */
        *consumerLink = returnSyncBlock;

        /* Create limiter */
        NvSciStreamBlock limiterBlock;
        if (!createLimiter(&limiterBlock, consOpts->useLimiter)) {
            return 0;
        }

        /* Connect to incoming consumer chain */
        if (NvSciError_Success !=
            NvSciStreamBlockConnect(limiterBlock, *consumerLink)) {
            printf("Failed to connect limiter to consumer chain\n");
            return 0;
        }

        /* Limiter is new end of chain */
        *consumerLink = limiterBlock;
    }

    return 1;
}

/* Set up IPC from producer to consumer */
static int32_t setupProdToConsIPC(
    NvSciStreamBlock* consumerLink,
    ConsumerOptions*  consOpts,
    bool              useExtEventService)
{
    if (!consOpts->c2cMode) {
        /* Create IPC block */
        if (!createIpcSrc(consumerLink,
                          consOpts->srcChannel,
                          useExtEventService)) {
            return 0;
        }
    } else {
        /* Create a queue for C2C src block */
        NvSciStreamBlock queueBlock;
        if (!createQueue(&queueBlock, consOpts->c2cSrcUseMailbox)) {
            return 0;
        }

        /* Create C2C block */
        if (!createC2cSrc(consumerLink, consOpts->srcChannel, queueBlock)) {
            return 0;
        }

        /* If mailbox is used with C2CSrc, then create presentSync block */
        if (1U == consOpts->c2cSrcUseMailbox) {
            NvSciStreamBlock presentSyncBlock;
            if (!createPresentSync(&presentSyncBlock)) {
                return 0;
            }

            if (NvSciError_Success !=
                NvSciStreamBlockConnect(presentSyncBlock, *consumerLink)) {
                printf("Failed to connect PresentSync to consumer chain\n");
                return 0;
            }

            /* PresentSync is new end of chain */
            *consumerLink = presentSyncBlock;
        }
    }
    return 1;
}

/* Set up IPC from consumer to producer */
static int32_t setupConsToProdIPC(
    NvSciStreamBlock* producerLink,
    ConsumerOptions*  consOpts,
    bool              useExtEventService)
{
    if (!consOpts->c2cMode) {
        /* Create IPC block */
        if (!createIpcDst(producerLink,
                          consOpts->dstChannel,
                          useExtEventService)) {
            return 0;
        }
    } else {
        /* Create a pool for C2C dst block */
        NvSciStreamBlock poolBlock;
        if (!createPool(&poolBlock, consOpts->c2cDstNumPacket, true)) {
            return 0;
        }
        /* Create C2C block */
        if (!createC2cDst(producerLink, consOpts->dstChannel, poolBlock)) {
            return 0;
        }
    }
    return 1;
}

/*
 * Main application function.
 *   As per standards, return of 0 indicates success and anything
 *   else is failure.
 */
int main(int argc, char *argv[])
{
    uint32_t i;
    int ret = 0;

    /* Initialize parameters */
    uint32_t badParam = 0U;
    uint32_t multiProcess = 0U;
    uint32_t multiSOC = 0U;
    uint32_t eventOption = 0U;

    ProducerOptions prodOpts = {.resident=0U, .numConsumer=1U,
                                .numPacket=3U, .numFrames=32, .usecase=1};
    memset(consOpts, 0, sizeof(consOpts));
    memset(&opts, 0, sizeof(CommonOptions));
    memset(ipcEP, 0, sizeof(Endpoint));

    /* Parse command line */
    int32_t opt;
    while ((opt = getopt(argc, argv, "m:n:r:f:l:q:k:e:ELs:u:ipc:P:C:F:Q:")) != EOF) {
        switch (opt) {
        case 'm': /* set number of consumers */
            prodOpts.numConsumer = atoi(optarg);
            opts.numConsumer = prodOpts.numConsumer;
            if ((prodOpts.numConsumer < 1U) ||
                (prodOpts.numConsumer > MAX_CONSUMERS)) {
                badParam = 1U;
            }
            for (i=0; i< MAX_CONSUMERS; i++) {
                sprintf(ipcEP[i].ipcChannel, "%s%d", ipcBaseName, 2*i+0);
                sprintf(ipcEP[i].ipcChannelForHandshake,
                        "%s%d", ipcBaseName, 2*i+8);
            }
            break;
        case 'r': /* set number of late consumers */
            opts.numLateConsumer = atoi(optarg);
            opts.lateAttach = true;
            if (opts.numLateConsumer > MAX_CONSUMERS) {
                badParam = 1U;
            }
            /* there must be atleast one early consumer */
            if ((prodOpts.numConsumer - opts.numLateConsumer) < 1U) {
                badParam = 1U;
            }
            prodOpts.numFrames = 100000;
            break;
        case 'f': /* set number of packets */
            prodOpts.numPacket = atoi(optarg);
            if ((prodOpts.numPacket < 1U) ||
                (prodOpts.numPacket > MAX_PACKETS)) {
                badParam = 1U;
            }
            break;

        case 'k': /* use specified number of frames for indexed consumer */
            i = atoi(optarg);
            if (i >= MAX_CONSUMERS) {
                badParam = 1U;
            } else {
                consOpts[i].numFrames = atoi(argv[optind++]);
            }
            break;

        case 'n': /* use specified number of frames for producer */
            prodOpts.numFrames = atoi(optarg);
            break;

        case 'l': /* use limiter block for indexed consumer */
            i = atoi(optarg);
            if (i >= MAX_CONSUMERS) {
                badParam = 1U;
            } else {
                consOpts[i].useLimiter = atoi(argv[optind++]);
            }
            break;
        case 'q': /* use specified queue for indexed consumer */
            i = atoi(optarg);
            if (i >= MAX_CONSUMERS) {
                badParam = 1U;
            } else {
                char t = argv[optind++][0];
                if (t == 'm') {
                    consOpts[i].useMailbox = 1U;
                } else if (t == 'f') {
                    consOpts[i].useMailbox = 0U;
                } else {
                    badParam = 1U;
                }
            }
            break;
        case 'e': /* set event handling mechanism */
            if (optarg[0] == 's') {
                eventOption = 0U;
            } else if (optarg[0] == 't') {
                eventOption = 1U;
            } else {
                badParam = 1U;
            }
            break;
        case 'E': /* set user-provided event service to handle IPC event */
            opts.useExtEventService = true;
            break;

        case 'L': /* Indicates late/reattaching of a consumer connection */
            opts.lateAttach = true;
            break;

        case 's': /* set Image Color Format type */
            if (optarg[0] == 'r') {
                opts.yuv = false;
            } else if (optarg[0] == 'y') {
                opts.yuv = true;
            } else {
                badParam = 1U;
            }
            break;
        case 'u': /* set use case */
            i = atoi(optarg);
            prodOpts.usecase = i;
            if (i == 1) {
                createProducer = createProducer_Usecase1;
                createConsumer = createConsumer_Usecase1;
                createPool     = createPool_Common;
            }
#if (NV_SUPPORT_NVMEDIA == 1)
            else if (i == 2) {
                createProducer = createProducer_Usecase2;
                createConsumer = createConsumer_Usecase2;
                createPool     = createPool_Common;

            }
#endif
#ifndef NV_SUPPORT_DESKTOP
            else if (i == 3) {
                createProducer = createProducer_Usecase3;
                createConsumer = createConsumer_Usecase3;
                createPool     = createPool_Usecase3;
            }
#endif
            else {
                badParam = 1U;
            }
            break;
        case 'i':
            opts.endInfo = true;
            break;

        /* For inter - process operation */

        case 'p': /* set producer resident */
            prodOpts.resident = 1U;
            multiProcess = 1U;
            break;
        case 'c': /* set consumer resident */
            i = atoi(optarg);
            if (i >= MAX_CONSUMERS) {
                badParam = 1U;
            } else {
                consOpts[i].resident = 1U;
                multiProcess = 1U;
            }
            break;

        /* For inter - chip (C2C) operation */

        case 'P': /* set ipc endpoint for C2C */
            i = atoi(optarg);
            if (i >= MAX_CONSUMERS) {
                badParam = 1U;
            } else {
                /* Ipc channel used to communicate with this C2C consumer */
                strcpy(consOpts[i].srcChannel, argv[optind++]);
                strcpy(ipcEP[i].c2cChannel, consOpts[i].srcChannel);
                consOpts[i].c2cMode = 1U;
                prodOpts.resident = 1U;
                multiProcess = 1U;
                multiSOC = 1U;
                opts.c2cMode = true;
            }
            break;
        case 'C': /* set C2C mode */
            i = atoi(optarg);
            if (i >= MAX_CONSUMERS) {
                badParam = 1U;
            } else {
                /* Ipc channel name used by this C2C consumer */
                strcpy(consOpts[i].dstChannel, argv[optind++]);
                consOpts[i].c2cMode = 1U;
                multiProcess = 1U;
                multiSOC = 1U;
                if (consOpts[i].c2cDstNumPacket == 0U) {
                    /* default packet size 3 if not set already */
                    consOpts[i].c2cDstNumPacket = 3U;
                }
            }
            break;
        case 'F': /* set number of packets for C2C Dst of indexed consumer */
            i = atoi(optarg);
            if (i >= MAX_CONSUMERS) {
                badParam = 1U;
            } else {
                consOpts[i].c2cDstNumPacket = atoi(argv[optind++]);
                if ((consOpts[i].c2cDstNumPacket < 1U) ||
                    (consOpts[i].c2cDstNumPacket > MAX_PACKETS)) {
                    badParam = 1U;
                }
            }
            break;
        case 'Q': /* use specified queue for C2C Src of indexed consumer */
            i = atoi(optarg);
            if (i >= MAX_CONSUMERS) {
                badParam = 1U;
            } else {
                char t = argv[optind++][0];
                if (t == 'm') {
                    consOpts[i].c2cSrcUseMailbox = 1U;
                } else if (t == 'f') {
                    consOpts[i].c2cSrcUseMailbox = 0U;
                } else {
                    badParam = 1U;
                }
            }
            break;
        default:
            badParam = 1U;
            break;
        }
    }

    /* Handle parsing failure */
    if (badParam) {
        print_usage(argv[0]);
        return 1;
    }

    /* Initialize streamDone */
    atomic_init(&streamDone, 0);

    if (opts.useExtEventService && (opts.numLateConsumer > 0U)) {
        /* Using external event service for late/re-attach usecase
        * is not supported for now.
        */
        return 1;
    }

    if (opts.useExtEventService && (eventOption == 1U)) {
        /* Using external event service for internal ipc I/O messages
         *  not supported with threading model in this sample app */
        return 1;
    }


    if ((prodOpts.usecase > 1U) && (opts.numLateConsumer > 0U)) {
        /* late/re-attach usecase is not supported except for usecase1
           in this sample app. */
        return 1;
    }

    /* Check validity of the combination C2C & non-C2C consumers */
    for (i=0U; i<MAX_CONSUMERS; ++i) {
        if (prodOpts.resident) {
            /* C2C consumer cannot be in the same process as producer */
            if (consOpts[i].resident && consOpts[i].c2cMode) {
                return 1;
            }
        } else {
            /* There is C2C consumer in this process,
            * can't have non-C2C ones
            */
            if (multiSOC && consOpts[i].resident) {
                return 1;
            }

            /* Now make consumer resident if C2C */
            if (consOpts[i].c2cMode) {
                consOpts[i].resident = 1U;
            }
        }
    }

    /* Fill in other options based on those specified */
    if (!multiProcess) {
        /* If no multi-process option specified, everything is resident */
        prodOpts.resident = 1U;
        for (i=0U; i<prodOpts.numConsumer; ++i) {
            consOpts[i].resident = 1U;
        }
    } else {
        /* If not in producer process, will just loop over full list */
        if (!prodOpts.resident) {
            prodOpts.numConsumer = MAX_CONSUMERS;
        }
        /* Channel names are derived from base and index */
        for (i=0U; i<prodOpts.numConsumer; ++i) {
            if (!consOpts[i].c2cMode) {
                sprintf(consOpts[i].srcChannel, "%s%d", ipcBaseName, 2*i+0);
                sprintf(consOpts[i].ipcChannelForHandshake,
                        "%s%d", ipcBaseName, 2*i+9);
                sprintf(consOpts[i].dstChannel, "%s%d", ipcBaseName, 2*i+1);
            }
        }
    }

    /* Select and initialize event-handling based on chosen method */
    eventFuncs = eventOption ? &eventFuncs_Threads : &eventFuncs_Service;
    if (!eventFuncs->init() != 0) {
        return 1;
    }

    /*
     * Initialize NvSci libraries
     */
    if (NvSciError_Success != NvSciSyncModuleOpen(&sciSyncModule)) {
        printf("Unable to open NvSciSync module\n");
    }
    if (NvSciError_Success != NvSciBufModuleOpen(&sciBufModule)) {
        printf("Unable to open NvSciBuf module\n");
    }
    if (NvSciError_Success != NvSciIpcInit()) {
        printf("Unable to initialize NvSciIpc\n");
    }

    /*
     * If producer is resident, create producer block chain and attach
     *  all consumers.
     */
    if (prodOpts.resident) {

        /* Set up producer chain (up through any multicast block) */
        NvSciStreamBlock producerLink;
        if (!setupProducerChain(&producerLink, &prodOpts)) {
            return 1;
        }

        /*
         * For each consumer, either set up the consumer chain or create
         *  the IPC block to communicate with it, depending on whether the
         *  consumer is resident.
         */
        for (i=0U; i<(prodOpts.numConsumer - opts.numLateConsumer); ++i) {

            /* Create consumer or IPC to consumer */
            NvSciStreamBlock consumerLink;
            if (consOpts[i].resident) {
                if (!setupConsumerChain(&consumerLink, &consOpts[i], i)) {
                    return 1;
                }
            } else {
                if (!setupProdToConsIPC(&consumerLink,
                                        &consOpts[i],
                                        opts.useExtEventService)) {
                    return 1;
                }
            }
             /* When Late-/reattach usecase is selected, a returnSync block
             * is connected to the consumer chain to ensure proper fence waits
             * during consumer disconnect and reconnect.
             */
             if (opts.numLateConsumer > 0U) {
                NvSciStreamBlock returnSyncBlock;
                if (!createReturnSync(&returnSyncBlock)) {
                    return 1;
                }

                /* Connect to incoming consumer chain */
               if (NvSciError_Success !=
                       NvSciStreamBlockConnect(returnSyncBlock, consumerLink)) {
                   printf("Failed to connect returnSyncBlock to consumer chain\n");
                   return 1;
               }
               consumerLink =  returnSyncBlock;
            }

            /* Add any other options (e.g. limiter) for this branch */
            if (!setupBranchOptions(&consumerLink, &consOpts[i])) {
                return 1;
            }

            /* Attach to producer chain */
            if (NvSciError_Success !=
                NvSciStreamBlockConnect(producerLink, consumerLink)) {
                printf("Failed to connect consumer %d to producer\n", i);
                return 1;
            }

        }
        if (opts.numLateConsumer > 0U) {
            NvSciError err;
            err = NvSciStreamBlockSetupStatusSet(multicastBlock,
                                                 NvSciStreamSetup_Connect,
                                                 true);
            if (err != NvSciError_Success) {
                printf("Attaching a late consumer connection failed=%x\n", err);
                return 1;
            }
        }
    }

    /*
     * Otherwise, create any consumer chains resident in this process,
     *   and connect with IPC back to the producer process.
     */
    else {

        for (i=0U; i<prodOpts.numConsumer; ++i) {
            if (consOpts[i].resident) {
                if (!consOpts[i].c2cMode && opts.lateAttach) {
                    if (true != openIpcEndpoint(i, true)) {
                        printf("Consumer failed to open endpoint \
                            needed for handshake\n");
                        return 1;
                    }
                    if(true != waitForIpcConnection(i, true)) {
                        printf("Consumer Connection failed\n");
                        return 1;
                    }
                }

                /* Create consumer */
                NvSciStreamBlock consumerLink;
                if (!setupConsumerChain(&consumerLink, &consOpts[i], i)) {
                    return 1;
                }

                /* Create IPC block */
                NvSciStreamBlock producerLink;
                if (!setupConsToProdIPC(&producerLink,
                                        &consOpts[i],
                                        opts.useExtEventService)) {
                    return 1;
                }

                /* Connect blocks */
                if (NvSciError_Success !=
                    NvSciStreamBlockConnect(producerLink, consumerLink)) {
                    printf("Failed to connect consumer %d to producer\n", i);
                    return 1;
                }
            }
        }
    }

    /* Enter event loop(s) until all blocks are done */
    if (!eventFuncs->loop()) {
        ret = 1;
    }

    /* Wakeup the dispatch thread to terminate upon
    * stream disconnect
    */
    atomic_store(&streamDone, 1);
    pthread_cond_signal(&cond);

    /* Wait for dispatch thread to terminate */
    if (opts.numLateConsumer > 0U) {
        (void)pthread_join(dispatchThread, NULL);
    }

    if (sciBufModule != NULL) {
        NvSciBufModuleClose(sciBufModule);
        sciBufModule = NULL;
    }

    if (sciSyncModule != NULL) {
        NvSciSyncModuleClose(sciSyncModule);
        sciSyncModule = NULL;
    }

    /* Close the NvSciIpc endpoint */
    for (uint32_t i = 0U; i< MAX_CONSUMERS; i++) {
#if (QNX == 1)
        if (ipcEP[i].coid != 0) {
            (void)ConnectDetach_r(ipcEP[i].coid);
            ipcEP[i].coid = 0;
        }
        if (ipcEP[i].chid != 0) {
            (void)ChannelDestroy_r(ipcEP[i].chid);
            ipcEP[i].chid = 0;
        }
#endif
        if (ipcEP[i].ipcEndpoint) {
            if (NvSciError_Success !=
                NvSciIpcCloseEndpointSafe(ipcEP[i].ipcEndpoint, false)) {
                printf("Failed to close ipc endpoint\n");
            }
            ipcEP[i].ipcEndpoint = 0U;
        }
        if (ipcEP[i].c2cEndpoint) {
            if (NvSciError_Success !=
                NvSciIpcCloseEndpointSafe(ipcEP[i].c2cEndpoint, false)) {
                printf("Failed to close c2c endpoint\n");
            }
            ipcEP[i].c2cEndpoint = 0U;
        }
    }

    /* freeing the resources */
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    NvSciIpcDeinit();

    return ret;
}
