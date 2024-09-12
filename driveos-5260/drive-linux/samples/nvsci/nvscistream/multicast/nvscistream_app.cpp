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
    printf(" -n <the number of consumer(1~4)> : Producer will wait n consumer");
    printf(" -p <channel name>: producer channel names, separated by white space\n");
    printf(" or\n");
    printf(" -c <channel name> -q <0/1>: consumer channel name + queue type (0: Mailbox (default), 1: Fifo)\n");
    printf("Producer Example) %s -n <the_number_of_consumers> -p <prod_channel_list>\n\n", str);
    printf("Consumer Example) %s -c cons_channel -q 1\n\n", str);
}

int main(int argc, char *argv[])
{
    char chname[MAX_CONSUMERS][32];
    bool isConsumerChannel = false;
    int32_t qtypeOpt = 0;
    int optcnt = 0;
    int32_t opt;
    uint32_t nConsumer = 0U;
    NvSciIpcEndpoint endpoint[MAX_CONSUMERS] = {0U};

    while ((opt = getopt(argc, argv, "p:c:q:n:")) != EOF)
    {
        switch (opt) {
            case 'p':
                /* Producer IPC channel name */
                if (nConsumer == 0U) {
                    print_usage(argv[0]);
                    return -1;
                } else {
                    for (uint32_t i = 0; i < nConsumer; i++) {
                        strncpy(chname[i], argv[PROD_CHANNEL_LIST_BASE_INDEX+i], sizeof(chname[i]));
                    }
                }
                optcnt++;
                break;
            case 'c':
                /* Consumer IPC channel name */
                strncpy(chname[0], optarg, sizeof(chname));
                /* This is not multi thread consumer case
                 * so nConsumer should be 1 in consumer case */
                nConsumer = 1U;
                optcnt++;
                isConsumerChannel = true;
                break;
            case 'q':
                /* queue type */
                qtypeOpt = strtoul(optarg, NULL, 0);
                break;
            case 'n':
                /* how many consumer will connect to producer */
                nConsumer = strtoul(optarg, NULL, 0);
                break;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }

    // we necessarily need a channel name
    // and only for either producer or consumer
    if (isConsumerChannel == false &&
        (nConsumer < 1|| nConsumer > 4)) {
        // wrong the number of consumer.
        optcnt--;
    }

    if (optcnt != 1) {
        print_usage(argv[0]);
        return -1;
    }

    NvSciBufModule bufModule;
    CHECK_NVSCIERR(NvSciBufModuleOpen(&bufModule));
    NvSciSyncModule syncModule;
    CHECK_NVSCIERR(NvSciSyncModuleOpen(&syncModule));
    CHECK_NVSCIERR(NvSciIpcInit());

    if (isConsumerChannel) {
        CHECK_NVSCIERR(NvSciIpcOpenEndpoint(chname[0], &endpoint[0]));
        NvSciIpcResetEndpoint(endpoint[0]);
        NvScistreamSample::ConsumerChannel *consChannel = new NvScistreamSample::ConsumerChannel(bufModule, syncModule, endpoint[0]);
        if (consChannel) {
            //consChannel->initIpc(ipcChannelName);
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
        for (uint32_t i=0U; i < nConsumer; i++) {
            CHECK_NVSCIERR(NvSciIpcOpenEndpoint(chname[i], &endpoint[i]));
            NvSciIpcResetEndpoint(endpoint[i]);
        }
        NvScistreamSample::ProducerChannel *prodChannel = new NvScistreamSample::ProducerChannel(bufModule, syncModule, endpoint, nConsumer);
        if (prodChannel) {
            //prodChannel->initIpc(ipcChannelName);
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

    for(uint32_t i = 0U; i < nConsumer; i++) {
        if (endpoint[i] != 0U) {
            NvSciIpcCloseEndpoint(endpoint[i]);
            endpoint[i] = 0U;
        }
    }
    NvSciIpcDeinit();

    return 0;
}