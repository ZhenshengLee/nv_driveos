# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
CPPFLAGS = $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
LDFLAGS  = $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_LDFLAGS)

ifeq ($(NV_PLATFORM_OS),QNX)
CFLAGS += -DQNX=1
else
CFLAGS += -DQNX=0
endif

OBJ := main.o
OBJ += proc_asil.o
OBJ += proc_qm.o
OBJ += proc_qm_proxy.o
OBJ += block_consumer.o
OBJ += block_pool.o
OBJ += block_producer.o
OBJ += util.o

LDLIBS :=
LDLIBS += -lnvscistream
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvscievent
LDLIBS += -lnvsciipc
LDLIBS += -lnvscicommon
LDLIBS += -lcuda
ifeq ($(NV_PLATFORM_OS),QNX)
  LDLIBS += -lnvdvms_client
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

nvscistream_safety_sample_app: $(OBJ)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -rf $(OBJ) nvscistream_safety_sample_app
