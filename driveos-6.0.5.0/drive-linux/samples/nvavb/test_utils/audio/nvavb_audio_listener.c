/*
 * Copyright (c) 2015, NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <assert.h>
#include "nvavtp.h"
#include <dlfcn.h>
#include <signal.h>

#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <unistd.h>
#include <errno.h>
#include <net/if.h>
#include "raw_socket.h"

#include <sched.h>
#include <sys/time.h>

#define PACKET_SIZE 298
#define AVB "avb"
//#define DEBUG_DEPACK

volatile sig_atomic_t loop_status;
void breakLoop(int sig);

typedef struct
{
    NvAvtpContextHandle pHandle;
    ENvAvtpBool firstPacket;
    FILE *fout;
    U8 *payload;
    U32 payloadSize;
    U32 pktcnt;
    char interface[64];
} NvmAvbSinkData;

void process_avb_data(NvmAvbSinkData *args, U8 *packet);
void initialize_AVTP(void *pParam);
void clean_exit(void *pParam);

#ifdef DEBUG_DEPACK
static U64 time_of_day(void);

//! wrapper for gettimeofday()
//! used for debug purpose
static U64 time_of_day(void)
{
    U64 temp_time = 0;
    struct timeval timeStruct;
    gettimeofday(&timeStruct, NULL);

    temp_time = ((timeStruct.tv_sec)*1000000LL) + timeStruct.tv_usec;
    return temp_time;
}
#endif

//! sets input parameters for AVTP
//! \param pParam - Pointer to NvAvbSinkData
void initialize_AVTP(void* pParam)
{
    NvAvtpInputParams *pAvtpInpPrms;
    NvmAvbSinkData *priv_data = (NvmAvbSinkData *)pParam;

    pAvtpInpPrms = malloc(sizeof(NvAvtpInputParams));
    if (pAvtpInpPrms == NULL)
    {
        exit(1);
    }

    memset(pAvtpInpPrms, 0x0, sizeof(NvAvtpInputParams));
    strcpy(pAvtpInpPrms->interface,priv_data->interface);
    pAvtpInpPrms->bAvtpDepacketization = eNvAvtpTrue;
    pAvtpInpPrms->eDataType = eNvMpegts;
    NvAvtpInit (pAvtpInpPrms, &priv_data->pHandle);
    free(pAvtpInpPrms);
}

//! clean up memory before exiting
//! \param pParam - Pointer to NvAvbSinkData
void clean_exit(void* pParam)
{
    NvmAvbSinkData *priv_data = (NvmAvbSinkData *) pParam;
    NvAvtpDeinit(priv_data->pHandle);
    if(priv_data->payload != NULL)
    {
        free(priv_data->payload);
    }
    if(priv_data->fout != NULL)
    {
        fclose(priv_data->fout);
    }
    free(priv_data);
}

//! function to process AVB packets
//! \param packet - Pointer to packet data
void process_avb_data(NvmAvbSinkData *args, U8 *packet)
{
    NvmAvbSinkData *priv_data = (NvmAvbSinkData *) args;
    ENvAvtpSubHeaderType eAvtpSubHeaderType;
    NvAvtpContextHandle pHandle = priv_data->pHandle;
    U32 payloadSize = 0;
#ifdef DEBUG_DEPACK
    fprintf(stderr,"Got packet.\n");
#endif

    if (NvAvtpIs1722Packet((U8 *)packet))
    {
        NvAvtpParseAvtpPacket(pHandle, (U8 *)packet, &eAvtpSubHeaderType);
        if(priv_data->firstPacket == eNvAvtpTrue)
        {
            priv_data->fout = fopen("out.pcm","w");
            U64 u64StreamId = NvAvtpGetStreamId(pHandle, (U8 *)packet);
            printf("Received stream id: %llx\n", u64StreamId);

            if(eAvtpSubHeaderType == eNvAudio)
            {
                printf("IEC 61883-6 Audio AVTP stream\n");
            }
            NvAvtpGetAudioDataPayloadSize(pHandle, (U8 *)packet, &priv_data->payloadSize);
            printf("Stream Length %d\n", priv_data->payloadSize);

            priv_data->payload = (U8 *)malloc(priv_data->payloadSize * sizeof(U8));
            priv_data->firstPacket = eNvAvtpFalse;
        }
        payloadSize = priv_data->payloadSize;
        NvAvtpGetAudioDataPayloadSize(pHandle, (U8 *)packet, &priv_data->payloadSize);
        if (payloadSize < priv_data->payloadSize)
        {
            free(priv_data->payload);
            priv_data->payload = (U8 *) malloc(priv_data->payloadSize * sizeof(U8));
        }
        NvAvtpExtractDataPayload(pHandle, (U8 *)packet, priv_data->payload);
        fwrite(priv_data->payload, priv_data->payloadSize, 1, priv_data->fout);
        priv_data->pktcnt++;
    }
    else
    {
        //Need to add filter to avoid it
        fprintf(stderr, "Not an AVTP packet - skipped\n");
    }

}

int main(int argc, char *argv[])
{

    struct sched_param sParam;
    NvmAvbSinkData *priv_data = NULL;
    char interface[64];
    S32 i;
    loop_status = 1;

    /*setting thread-priority as real-time*/
    memset(&sParam, 0, sizeof(struct sched_param));
    sParam.sched_priority = sched_get_priority_max(SCHED_RR);
    int retval = sched_setscheduler(0, SCHED_RR, &sParam);
    if (retval != 0)
    {
        fprintf(stderr, "%s", "Scheduling error.\n");
        exit(1);
    }

    /*setting signal handler*/
    signal(SIGINT, breakLoop);

    /*argument parsing*/
    if (argc >= 2)
    {
        for(i=0;i<argc;i++)
        {
            if (!strcmp (argv[i], "-i"))
            {
                strcpy (interface, argv[i + 1]);
            }
        }
    }
    else
    {
        strcpy (interface, "eth2");
    }

    /*initialization*/
    priv_data = (NvmAvbSinkData *)malloc(sizeof(NvmAvbSinkData));
    memset(priv_data, 0x00, sizeof(NvmAvbSinkData));
    priv_data->firstPacket = eNvAvtpTrue;
    strcpy(priv_data->interface, interface);

    char *type = "avb";
    if (0 == memcmp(type, AVB, sizeof(AVB)) )
    {
        initialize_AVTP(priv_data);
    }
    else
    {
        fprintf(stderr, "%s", "Invalid capture format\n");
    }

    /*raw socket setup*/
    int fd_read_socket;
    S32 bytesRead = 0;
    U8 buffer[2148];

    fd_read_socket = set_socket((char*)priv_data->interface);

    printf("----------\n");
    while (loop_status)
    {

#ifdef DEBUG_DEPACK
        U64 start_time = time_of_day();
#endif
        bytesRead = recvfrom(fd_read_socket,buffer,2148,0,NULL,NULL);

        if (bytesRead < 0)
        {
            break;
        }
        /*packet is too short*/
        if (bytesRead < 14)
        {
            perror("recvfrom():");
            printf("Bytes read: %d\n", bytesRead);
            printf("Incomplete packet (errno is %d)\n", errno);
            close(fd_read_socket);
            exit(1);
        }

        if (NvAvtpIs1722Packet((U8 *)buffer))
        {
            process_avb_data(priv_data, (U8*)&buffer);
        }

#ifdef DEBUG_DEPACK
        U64 end_time = time_of_day();
        printf("Processing time: %llu\n", end_time - start_time);
        printf("Packet size: %d\n", bytesRead);
#endif
     }

    close(fd_read_socket);
    clean_exit(priv_data);
    return 0;
}

//! signal handler
//! function to stop reading socket
void breakLoop(int sig)
{
    fprintf(stderr, "\n");
    loop_status = 0;
}
