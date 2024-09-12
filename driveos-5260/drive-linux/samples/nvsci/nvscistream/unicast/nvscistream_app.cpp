//
// NvSciStream Sample App.
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "channel_producer.h"
#include "channel_consumer.h"
#include "util.h"
#include <cstring>
#include <unistd.h>

static void print_usage(const char *str)
{
    printf("%s: Specify producer channel or consumer channel\n", str);
    printf(" -p <channel name>: producer channel name\n");
    printf(" or\n");
    printf(" -c <channel name> -q <0/1>: consumer channel name + queue type (0: Mailbox (default), 1: Fifo)\n");
    printf("Producer Example) %s -p prod_channel\n\n", str);
    printf("Consumer Example) %s -c cons_channel -q 1\n\n", str);
}

int main(int argc, char *argv[])
{
    char chname[32];
    bool isConsumerChannel = false;
    int32_t qtypeOpt = 0;
    int optcnt = 0;
    int32_t opt;
    NvSciIpcEndpoint endpoint = 0U;
    while ((opt = getopt(argc, argv, "p:c:q:")) != EOF)
    {
        switch (opt) {
            case 'p':
                /* Producer IPC channel name */
                strncpy(chname, optarg, sizeof(chname));
                optcnt++;
                break;
            case 'c':
                /* Consumer IPC channel name */
                strncpy(chname, optarg, sizeof(chname));
                optcnt++;
                isConsumerChannel = true;
                break;
            case 'q':
                /* queue type */
                qtypeOpt = strtoul(optarg, NULL, 0);
                break;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }

    // we necessarily need a channel name
    // and only for either producer or consumer
    if (optcnt != 1) {
        print_usage(argv[0]);
        return -1;
    }

    const char* ipcChannelName = chname;
    NvSciBufModule bufModule;
    CHECK_NVSCIERR(NvSciBufModuleOpen(&bufModule));
    NvSciSyncModule syncModule;
    CHECK_NVSCIERR(NvSciSyncModuleOpen(&syncModule));
    CHECK_NVSCIERR(NvSciIpcInit());
    CHECK_NVSCIERR(NvSciIpcOpenEndpoint(ipcChannelName, &endpoint));
    NvSciIpcResetEndpoint(endpoint);

    if (isConsumerChannel) {
        NvScistreamSample::ConsumerChannel *consChannel = new NvScistreamSample::ConsumerChannel(bufModule, syncModule, endpoint);
        if (consChannel) {
            typedef NvScistreamSample::QueueType QType;
            QType qType = (qtypeOpt == 1) ? QType::QueueType_FIFO
                                          : QType::QueueType_Mailbox;
            consChannel->createBlocks(qType);
            consChannel->connectToStream();
            consChannel->setupForStreaming();
            consChannel->runStream();
            delete consChannel;
        } else {
            LOG_ERR_EXIT("\nFailed to create Consumer channel\n");
        }
    } else {
        NvScistreamSample::ProducerChannel *prodChannel = new NvScistreamSample::ProducerChannel(bufModule, syncModule, endpoint);
        if (prodChannel) {
            prodChannel->createBlocks();
            prodChannel->connectToStream();
            prodChannel->setupForStreaming();
            prodChannel->runStream();
            delete prodChannel;
        } else {
            LOG_ERR_EXIT("\nFailed to create Producer channel\n");
        }
    }

    if (bufModule != nullptr)
    {
        NvSciBufModuleClose(bufModule);
        bufModule = nullptr;
    }

    if (syncModule != nullptr)
    {
        NvSciSyncModuleClose(syncModule);
        syncModule = nullptr;
    }
    if (endpoint != 0U) {
       NvSciIpcCloseEndpoint(endpoint);
       endpoint = 0U;
    }
    NvSciIpcDeinit();
    return 0;
}