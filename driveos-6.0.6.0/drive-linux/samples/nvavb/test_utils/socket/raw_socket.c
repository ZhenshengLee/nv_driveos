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
#include <unistd.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <net/if.h>
#include <arpa/inet.h>

#include "raw_socket.h"

void set_socket_params(int sock, struct packet_mreq* mr, struct sockaddr_ll* device)
{
    memset(mr, 0, sizeof(struct packet_mreq));
    mr->mr_ifindex = device->sll_ifindex;
    mr->mr_type = PACKET_MR_PROMISC;
    setsockopt(sock, SOL_PACKET, PACKET_ADD_MEMBERSHIP, mr, sizeof(struct packet_mreq));

    struct timeval timeout;
    timeout.tv_sec = 15;
    timeout.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
}

void get_socket_info(struct sockaddr_ll *device, char* interface)
{
    memset(device, 0, sizeof(struct sockaddr_ll));
    device->sll_family = PF_PACKET;
    device->sll_ifindex = if_nametoindex(interface);
}

int set_socket(char* interface)
{
    int sock = 0;

    if (interface == NULL)
    {
        goto cleanup;
    }

    sock = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sock < 0)
    {
        perror("socket");
        exit(1);
    }

    struct sockaddr_ll device;
    get_socket_info(&device, interface);
    bind(sock, (struct sockaddr*) &device, sizeof(device));

    struct packet_mreq mr;
    set_socket_params(sock, &mr, &device);

    return sock;

    cleanup:
        close(sock);
        return -1;
}
