# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

include ../../../../make/nvdefs.mk

CFLAGS   = $(NV_PLATFORM_OPT)     $(NV_PLATFORM_CFLAGS)
CPPFLAGS = $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS) -I$(NV_PLATFORM_DIR)/kernel/source/oss_src/nvidia/include/uapi
LDFLAGS  = $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_LDFLAGS)

ifeq ($(NV_PLATFORM_OS),QNX)
CPPFLAGS += -DQNX=1
else
CPPFLAGS += -DQNX=0
endif

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
ifeq ($(NV_PLATFORM_OS),QNX)
  LDLIBS += -lnvdvms_client
  LDLIBS += -lslog2
  LDLIBS += -lsocket
  LDLIBS += -lc++
  LDLIBS += -lnvtime2
  LDLIBS += -lnvos_s3_safety
else
  LDLIBS += -lpthread
  LDLIBS += -lrt
  LDLIBS += -ldl
  LDLIBS += -lstdc++
endif

test_nvscistream_perf: $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -rf $(OBJS) test_nvscistream_perf
