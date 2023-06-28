/*
 * Copyright (c) 2015, NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <linux/if_ether.h>
#include <linux/if_packet.h>

void set_socket_params(int sock, struct packet_mreq* mr, struct sockaddr_ll* device);
void get_socket_info(struct sockaddr_ll *device, char* interface);
int set_socket(char* interface);
