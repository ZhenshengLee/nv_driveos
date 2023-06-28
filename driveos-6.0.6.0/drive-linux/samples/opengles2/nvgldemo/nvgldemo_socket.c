/*
 * nvgldemo_socket.c
 *
 * Copyright (c) 2010-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//
// This file illustrates how to create an EGL Stream from a file descriptor.
//   In this case the fd is acquired from a socket.
//

#include "nvgldemo.h"

#if defined (__INTEGRITY)
#include <sys/uio.h>
#endif

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <errno.h>

int
NvGlDemoFdFromSocket(char const *name)
{
    struct sockaddr_un address = { 0 };
    struct msghdr msg = { 0 };
    struct iovec iov;
    struct cmsghdr *cmsg;
    char msg_buf[1];
    char ctl_buf[CMSG_SPACE(sizeof(int))];
    int sockfd = -1;
    int streamfd = -1;

    sockfd = socket(PF_UNIX, SOCK_STREAM, 0);
    if (sockfd == -1) {
        goto fail;
    }

    address.sun_family = AF_UNIX;
    STRNCPY(address.sun_path, name, sizeof(address.sun_path) - 1);

    while (connect(sockfd,
                   (const struct sockaddr *)&address,
                   sizeof(address))) {
        NvGlDemoLog("Waiting for server.\n");
        sleep(1);
    }

    iov.iov_base = msg_buf;
    iov.iov_len = sizeof(msg_buf);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = ctl_buf;
    msg.msg_controllen = sizeof(ctl_buf);

    if (recvmsg(sockfd, &msg, 0) <= 0) {
        goto fail;
    }

    cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == NULL
        || cmsg->cmsg_level != SOL_SOCKET
        || cmsg->cmsg_type != SCM_RIGHTS) {
        /* Probably connected to somebody else's socket. */
        goto fail;
    }

    MEMCPY(&streamfd, CMSG_DATA(cmsg), sizeof(int));
    (void)close(sockfd);

    return streamfd;

fail:
    if (sockfd != -1)
        (void)close(sockfd);
    return -1;
}

int
NvGlDemoCloseFd(int fd)
{
    return close(fd);
}
