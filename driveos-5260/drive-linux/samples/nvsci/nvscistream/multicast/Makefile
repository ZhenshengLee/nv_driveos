# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
OBJS += client_common.o
OBJS += cuda_consumer.o
OBJS += nvmedia_producer.o
OBJS += nvscistream_app.o
OBJS += packet_pool_manager.o

LDLIBS :=
LDLIBS += -lnvscistream
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvsciipc
LDLIBS += -lnvscicommon
LDLIBS += -lnvmedia
LDLIBS += -lcuda

nvscistream_multicast_sample: $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -rf $(OBJS) nvscistream_multicast_sample
