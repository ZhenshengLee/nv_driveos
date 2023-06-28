Rawstream NvSciIpcC2C Sample App - README

Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

NVIDIA Corporation and its licensors retain all intellectual property and
proprietary rights in and to this software, related documentation and any
modifications thereto. Any use, reproduction, disclosure or distribution
of this software and related documentation without an express license
agreement from NVIDIA Corporation is strictly prohibited.

---
# NvSciIpcC2C - NvStreams Rawstream NvSciIpcC2C Sample App

## Description

This directory contains a raw stream NvSciIpcC2C sample application.

## Build the application

The rawstream sample includes source code and a Makefile.
1. On the host system, navigate to the sample application directory:
> cd <top>/samples/nvsci/rawstream_c2c/

2. Build the sample application:
> make clean
> make

## Examples of how to run the sample application:
Please check the C2C Endpoint is created:
1. get INTER_CHIP c2c node informations
> cat /etc/nvsciipc.cfg

2. make sure the INTER_CHIP c2c node was created on /dev/*
> ls /dev/*

3. start the write sample to send on src c2c endpoint
> ./nvsciipc_write -c <C2C IpcSrc Endpoint> -T <timeout ms> -v -l <loop time>

4. start the read sample on peer soc to receive the data
> ./nvsciipc_read -c <C2C IpcDst Endpoint> -T <timeout ms> -v -l <loop time>

5. For more informations
  ./nvsciipc_write -h for help
  ./nvsciipc_read -h for help