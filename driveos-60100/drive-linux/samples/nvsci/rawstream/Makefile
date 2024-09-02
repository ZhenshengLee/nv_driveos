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

OBJS :=
OBJS += rawstream_main.o
OBJS += rawstream_producer.o
OBJS += rawstream_consumer.o
OBJS += rawstream_ipc_linux.o
OBJS += rawstream_cuda.o

LDLIBS :=
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvsciipc
LDLIBS += -lnvscicommon
LDLIBS += -lcuda
LDLIBS += -lpthread
LDLIBS += -lstdc++
LDLIBS += -L$(NV_PLATFORM_CUDA_TOOLKIT)/targets/aarch64-linux/lib/ -lcudart

rawstream: $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f rawstream *.o
