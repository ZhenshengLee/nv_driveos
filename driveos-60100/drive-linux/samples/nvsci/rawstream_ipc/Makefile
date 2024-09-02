# Copyright (c) 2020-2022 NVIDIA Corporation.  All Rights Reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and
# proprietary rights in and to this software and related documentation.  Any
# use, reproduction, disclosure or distribution of this software and related
# documentation without an express license agreement from NVIDIA Corporation
# is strictly prohibited.

CFLAGS   = $(NV_PLATFORM_OPT)     $(NV_PLATFORM_CFLAGS)
CXXFLAGS = $(NV_PLATFORM_OPT)     $(NV_PLATFORM_CXXFLAGS) -std=c++11
CPPFLAGS = $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
LDFLAGS  = $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_LDFLAGS)

include ../../../make/nvdefs.mk

LDLIBS :=
LDLIBS += -I ../../../include
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvsciipc
LDLIBS += -lnvscievent
LDLIBS += -lnvscicommon
LDLIBS += -lpthread
LDLIBS += -lstdc++
LDLIBS += -L$(NV_PLATFORM_CUDA_TOOLKIT)/targets/aarch64-linux/lib/

all: nvsciipc_write nvsciipc_read

nvsciipc_write: nvsciipc_write.c
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

nvsciipc_read: nvsciipc_read.c
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f nvsciipc_write
	rm -f nvsciipc_read
