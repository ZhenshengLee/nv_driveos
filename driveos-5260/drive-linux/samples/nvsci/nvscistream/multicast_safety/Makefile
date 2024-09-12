# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#

include ../../../../make/nvdefs.mk

CFLAGS   = $(NV_PLATFORM_OPT)     $(NV_PLATFORM_CFLAGS)
CXXFLAGS = $(NV_PLATFORM_OPT)     $(NV_PLATFORM_CXXFLAGS) -std=c++11
CPPFLAGS = $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
LDFLAGS  = $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_LDFLAGS)

OBJS :=
OBJS += channel_consumer.o
OBJS += channel_producer.o
OBJS += cuda_common.o
OBJS += cuda_consumer.o
OBJS += cuda_producer.o
OBJS += nvscistream_app.o
OBJS += packet_pool_manager.o

LDLIBS :=
LDLIBS += -lnvscistream
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvsciipc
LDLIBS += -lnvscicommon
LDLIBS += -lcuda
LDLIBS += -lpthread
LDLIBS += -L$(NV_PLATFORM_CUDA_TOOLKIT)/targets/aarch64-linux/lib/ -lcudart

nvscistream_multicast_safety_sample: $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f nvscistream_multicast_safety_sample *.o
