/* NvSciStream Event Loop Driven Sample App - main application
 *
 * Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto. Any use, reproduction, disclosure or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 */

/*
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
#include "nvscisync.h"
#include "nvscibuf.h"
#include "nvsciipc.h"
#include "nvscistream.h"
#include "event_loop.h"
#include "block_info.h"

/* Base name for all IPC channels */
static const char ipcBaseName[] = "nvscistream_";

/* Event handling function table */
EventFuncs const* eventFuncs = NULL;

/* Top level use-case setup function pointers */
int32_t (*createProducer)(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool) = createProducer_Usecase1;
int32_t (*createConsumer)(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  queue,
    uint32_t          index) = createConsumer_Usecase1;

/* NvSci modules */
NvSciSyncModule sciSyncModule;
NvSciBufModule  sciBufModule;

/* NvSciIpc Endpoint */
NvSciIpcEndpoint ipcEndpoint = 0U;

/* Common options for all blocks */
CommonOptions opts;

/* Options for producer */
typedef struct {
    uint32_t resident;
    uint32_t numConsumer;
    uint32_t numPacket;
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
} ConsumerOptions;

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
    printf("   -u <index> [default 1]\n");
    printf("       use case (must be same for all processes)\n");
    printf("       1 : CUDA (rt) producer to CUDA (rt) consumer\n");
#if (NV_SUPPORT_NVMEDIA == 1)
    printf("       2 : NvMedia producer to CUDA (rt) consumer\n");
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
    if (!createProducer(&producerBlock, poolBlock)) {
        return 0;
    }

    /* If multicast required, add the block. */
    if (prodOpts->numConsumer > 1) {

        /* Create multicast block */
        NvSciStreamBlock multicastBlock;
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
    if (!createConsumer(&consumerBlock, queueBlock, index)) {
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
    ConsumerOptions*  consOpts)
{
    if (!consOpts->c2cMode) {
        /* Create IPC block */
        if (!createIpcSrc(consumerLink, consOpts->srcChannel)) {
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
    ConsumerOptions*  consOpts)
{
    if (!consOpts->c2cMode) {
        /* Create IPC block */
        if (!createIpcDst(producerLink, consOpts->dstChannel)) {
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
    ProducerOptions prodOpts = {.resident=0U, .numConsumer=1U, .numPacket=3U};
    ConsumerOptions consOpts[MAX_CONSUMERS];
    memset(consOpts, 0, sizeof(consOpts));
    memset(&opts, 0, sizeof(CommonOptions));

    /* Parse command line */
    int32_t opt;
    while ((opt = getopt(argc, argv, "m:f:l:q:e:u:ipc:P:C:F:Q:")) != EOF) {
        switch (opt) {
        case 'm': /* set number of consumers */
            prodOpts.numConsumer = atoi(optarg);
            if ((prodOpts.numConsumer < 1U) ||
                (prodOpts.numConsumer > MAX_CONSUMERS)) {
                badParam = 1U;
            }
            break;
        case 'f': /* set number of packets */
            prodOpts.numPacket = atoi(optarg);
            if ((prodOpts.numPacket < 1U) ||
                (prodOpts.numPacket > MAX_PACKETS)) {
                badParam = 1U;
            }
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
        case 'u': /* set use case */
            i = atoi(optarg);
            if (i == 1) {
                createProducer = createProducer_Usecase1;
                createConsumer = createConsumer_Usecase1;
            }
#if (NV_SUPPORT_NVMEDIA == 1)
            else if (i == 2) {
                createProducer = createProducer_Usecase2;
                createConsumer = createConsumer_Usecase2;
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
                consOpts[i].c2cMode = 1U;
                prodOpts.resident = 1U;
                multiProcess = 1U;
                multiSOC = 1U;
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
        for (i=0U; i<prodOpts.numConsumer; ++i) {

            /* Create consumer or IPC to consumer */
            NvSciStreamBlock consumerLink;
            if (consOpts[i].resident) {
                if (!setupConsumerChain(&consumerLink, &consOpts[i], i)) {
                    return 1;
                }
            } else {
                if (!setupProdToConsIPC(&consumerLink, &consOpts[i])) {
                    return 1;
                }
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
    }

    /*
     * Otherwise, create any consumer chains resident in this process,
     *   and connect with IPC back to the producer process.
     */
    else {
        for (i=0U; i<prodOpts.numConsumer; ++i) {
            if (consOpts[i].resident) {

                /* Create consumer */
                NvSciStreamBlock consumerLink;
                if (!setupConsumerChain(&consumerLink, &consOpts[i], i)) {
                    return 1;
                }

                /* Create IPC block */
                NvSciStreamBlock producerLink;
                if (!setupConsToProdIPC(&producerLink, &consOpts[i])) {
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

    if (sciBufModule != NULL) {
        NvSciBufModuleClose(sciBufModule);
        sciBufModule = NULL;
    }

    if (sciSyncModule != NULL) {
        NvSciSyncModuleClose(sciSyncModule);
        sciSyncModule = NULL;
    }

    /* Close the NvSciIpc endpoint */
    if (ipcEndpoint) {
        if (NvSciError_Success !=
            NvSciIpcCloseEndpointSafe(ipcEndpoint, false)) {
            printf("Failed to close ipc endpoint\n");
        }
        ipcEndpoint = 0U;
    }

    NvSciIpcDeinit();

    return ret;
}
