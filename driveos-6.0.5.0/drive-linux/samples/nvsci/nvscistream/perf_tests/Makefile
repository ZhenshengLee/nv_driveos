# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#

include ../../../../make/nvdefs.mk

CFLAGS   = $(NV_PLATFORM_OPT)     $(NV_PLATFORM_CFLAGS)
CPPFLAGS = $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
LDFLAGS  = $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_LDFLAGS)

OBJS :=
OBJS += main.o
OBJS += perfclient.o
OBJS += perfconsumer.o
OBJS += perfproducer.o
OBJS += perftest.o
OBJS += poolhandler.o

LDLIBS :=
LDLIBS += -lnvscistream
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvsciipc
LDLIBS += -lnvplayfair
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

test_nvscistream_perf: $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -rf $(OBJS) test_nvscistream_perf
