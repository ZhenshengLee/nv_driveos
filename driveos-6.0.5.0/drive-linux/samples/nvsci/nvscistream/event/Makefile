# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#

include ../../../../make/nvdefs.mk

CFLAGS   = $(NV_PLATFORM_OPT)     $(NV_PLATFORM_CFLAGS) -I../../../../include/nvmedia_6x/
CPPFLAGS = $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
LDFLAGS  = $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_LDFLAGS)

ifeq ($(NV_PLATFORM_OS),QNX)
CFLAGS += -DQNX=1
else
CFLAGS += -DQNX=0
endif

CFLAGS += -DNV_SUPPORT_NVMEDIA=1

OBJS :=
OBJS += main.o
OBJS += event_loop_service.o
OBJS += event_loop_threads.o
OBJS += block_c2c.o
OBJS += block_common.o
OBJS += block_consumer_uc1.o
OBJS += block_consumer_uc2.o
OBJS += block_ipc.o
OBJS += block_limiter.o
OBJS += block_multicast.o
OBJS += block_pool.o
OBJS += block_presentsync.o
OBJS += block_producer_uc1.o
OBJS += block_producer_uc2.o
OBJS += block_queue.o
OBJS += block_returnsync.o
OBJS += util.o

LDLIBS :=
LDLIBS += -lnvscistream
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvscievent
LDLIBS += -lnvsciipc
LDLIBS += -lnvscicommon
LDLIBS += -lcuda
ifeq ($(NV_PLATFORM_OS),QNX)
  LDLIBS += $(NV_PLATFORM_CUDA_LIB)/libcudart_static.a
  LDLIBS += -lslog2
  LDLIBS += -lsocket
  LDLIBS += -lc++
else
  LDLIBS += -L$(NV_PLATFORM_CUDA_TOOLKIT)/targets/aarch64-linux/lib/ -lcudart
  LDLIBS += -lpthread
  LDLIBS += -lrt
  LDLIBS += -ldl
  LDLIBS += -lstdc++
endif
LDLIBS += -lnvmedia2d

nvscistream_event_sample: $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -rf $(OBJS) nvscistream_event_sample
