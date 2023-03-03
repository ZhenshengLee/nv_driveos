Rawstream Sample App - README

Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

NVIDIA Corporation and its licensors retain all intellectual property and
proprietary rights in and to this software, related documentation and any
modifications thereto. Any use, reproduction, disclosure or distribution
of this software and related documentation without an express license
agreement from NVIDIA Corporation is strictly prohibited.

---
# rawstream - NvStreams Rawstream Sample App

## Description

This directory contains a raw stream sample application using NvSciBuf,
NvSciSync and NvSciIpc.


## Build the application

The rawstream sample includes source code and a Makefile.
1. On the host system, navigate to the sample application directory:

       $ cd <top>/samples/nvsci/rawstream/

2. Build the sample application:

       $ make clean
       $ make


## Examples of how to run the sample application:

    $ sudo ./rawstream -p &
    $ sudo ./rawstream -c
