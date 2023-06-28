/*
 * Copyright (c) 2018, NVIDIA Corporation.  All rights reserved.
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
#include "nvavtp.h"
#include "pcap.h"
#include <dlfcn.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/if.h>
#include <netpacket/packet.h>
#include <netinet/in.h>
#include <net/ethernet.h>
#include <fcntl.h>
#include <unistd.h>


struct ifr_datastruct_1722
{
    U32 txred_len;
    U32 pkt_len;
    U8  data[1500];
};

/*lookup table for libpcap*/
struct pcap_dl {
    void* handle;
    pcap_t* (*pcap_open_offline_t) (const char *, char *);
    const u_char* (*pcap_next_t) (pcap_t *, struct pcap_pkthdr *);
    const char *dl_error;
} dlpcap;
/*unit to hold common data*/
typedef struct {
    NvAvtpContextHandle pHandle;
    NvAvtp1722AAFParams  *pAvtp1722AAFParameters;
    U32 pktcnt;
} NvAvbData;

static U8 init_dl(void);
static U8 deinit_dl (void);
void initialize_AVTP(void* pParam);
void clean_exit(void* pParam);

U8 DEST_ADDR[] = { 0x91, 0xE0, 0xF0, 0x00, 0x0e, 0x80 };
//U8 DEST_ADDR[] = { 0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };

int main(int argc, char *argv[])
{
    struct sched_param sParam;
    int pkts =0;

    /*setting thread-priority as real-time*/
    memset(&sParam, 0, sizeof(struct sched_param));
    sParam.sched_priority = sched_get_priority_max(SCHED_RR);
    int retval = sched_setscheduler(0, SCHED_RR, &sParam);
    if (retval != 0)
    {
        fprintf(stderr, "%s", "Scheduling error.\n");
        exit(1);
    }

    pcap_t* pcap;
    int lsock_aaf;
    struct ifreq ifr;
    struct ifr_datastruct_1722 pkt_1722;
    struct sockaddr_ll socket_address;
    const unsigned char* packet;
    char errbuf[PCAP_ERRBUF_SIZE];
    struct pcap_pkthdr pcap_header;
    U8 result;
    U32 rc;
    NvAvbData* data_block;

    ++argv; --argc;

    if(argc != 1)
    {
        fprintf(stderr, "program requires path to packet dump");
        exit(1);
    }

    result = init_dl();
    if (result == 0)
    {
        fprintf (stderr, "Error while opening libpcap ");
        exit(1);
    }

    lsock_aaf = socket( AF_PACKET, SOCK_RAW, IPPROTO_RAW );
    //lsock_audio = socket( PF_PACKET, SOCK_DGRAM, 0 );
    if( lsock_aaf < 0 )
    {
        perror("Socket error");
        exit(1);
    }

    pcap = dlpcap.pcap_open_offline_t(argv[0], errbuf);
    if(pcap == NULL)
    {
        fprintf(stderr, "error readping pcap file %s\n", errbuf);
        exit(1);
    }

    data_block = (NvAvbData*) malloc(sizeof(NvAvbData));
    memset(data_block, 0x0, sizeof(NvAvbData));
    initialize_AVTP(data_block);

    while( (packet = dlpcap.pcap_next_t(pcap, &pcap_header)) != NULL)
    {
        /* Init tx parameters */
        memset(&ifr, 0, sizeof(ifr));
        strcpy(ifr.ifr_name, "eth0");

        ifr.ifr_data = (void *) (&pkt_1722 + 8);
        if ((rc = ioctl(lsock_aaf, SIOCGIFINDEX, &ifr)) < 0)
        {
            exit(1);
        }

        /* Index of the network device */
        socket_address.sll_ifindex = ifr.ifr_ifindex;
        /* Address length*/
        socket_address.sll_halen = ETH_ALEN;
        memcpy(socket_address.sll_addr,  DEST_ADDR, sizeof(DEST_ADDR));
        memcpy(pkt_1722.data, (U8 *)packet, pcap_header.len);
        /* Init tx parameters */
        pkt_1722.txred_len = 0;
        pkt_1722.pkt_len = pcap_header.len;

        sendto(lsock_aaf, pkt_1722.data,  pkt_1722.pkt_len, 0,
          (struct sockaddr*)&socket_address, sizeof(struct sockaddr_ll));

        usleep(10000);
        pkts++;
    }
    printf("packets sent %d\n", pkts);
    clean_exit(data_block);
    deinit_dl();

    return 0;
}

/*sets input parameters for AVTP*/
void initialize_AVTP(void* pParam)
{
    NvAvtpInputParams *pAvtpInpPrms;
    NvAvbData *data_block = (NvAvbData *)pParam;

    pAvtpInpPrms = malloc(sizeof(NvAvtpInputParams));
    if (pAvtpInpPrms == NULL)
    {
        exit(1);
    }

    memset(pAvtpInpPrms, 0x0, sizeof(NvAvtpInputParams));
    pAvtpInpPrms->bAvtpDepacketization = eNvAvtpFalse;
    pAvtpInpPrms->eDataType = eNvAAF;
    NvAvtpInit (pAvtpInpPrms, &data_block->pHandle);
    free(pAvtpInpPrms);
}

/*clean up memory before exiting*/
void clean_exit(void* pointer)
{
    NvAvbData *data_block = (NvAvbData *) pointer;
    NvAvtpDeinit(data_block->pHandle);
    free(data_block);
}

static
U8 init_dl (void)
{
   if (!dlpcap.handle) {
       dlpcap.handle = dlopen ("libpcap.so", RTLD_LAZY);
       dlpcap.dl_error = dlerror ();
       if (dlpcap.dl_error) {
           printf ("Cannot open libpcap. %s",dlpcap.dl_error);
           dlclose (dlpcap.handle);
           dlpcap.handle = NULL;
           return 0;
       }
   }

   dlpcap.pcap_open_offline_t = dlsym (dlpcap.handle, "pcap_open_offline");
   dlpcap.dl_error = dlerror ();
   if (dlpcap.dl_error) {
       printf ("Cannot get pcap_open_offline. %s",dlpcap.dl_error);
      dlpcap.pcap_open_offline_t = NULL;
       return 0;
   }

   dlpcap.pcap_next_t = dlsym (dlpcap.handle, "pcap_next");
   dlpcap.dl_error = dlerror ();
   if (dlpcap.dl_error) {
       printf ("Cannot get pcap_next. %s",dlpcap.dl_error);
       dlpcap.pcap_next_t = NULL;
       return 0;
   }
   return 1;
}

static
U8 deinit_dl (void)
{
    if (dlpcap.handle) {
       dlclose (dlpcap.handle);
       dlpcap.handle = NULL;
    }
    if (dlpcap.pcap_open_offline_t) {
       dlpcap.pcap_open_offline_t = NULL;
    }
    if (dlpcap.pcap_next_t) {
       dlpcap.pcap_next_t = NULL;
    }
    return 1;
}
