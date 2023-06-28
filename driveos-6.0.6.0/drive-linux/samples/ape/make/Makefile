# Copyright (c) 2016-2018, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ./../../../make/nvapedefs.mk


PLUGIN = adsp
TARGET_ELF = $(addprefix ./target/,$(PLUGIN).elf)
TARGET_BIN = $(addprefix ./target/,$(PLUGIN).bin)
TARGET_DBG = $(addprefix ./target/,$(PLUGIN).debug.lst)
TARGET_SIZ = $(addprefix ./target/,$(PLUGIN).size)
TARGET_SYM = $(addprefix ./target/,$(PLUGIN).sym)
TARGET_LST = $(addprefix ./target/,$(PLUGIN).lst)
TARGET_STK = $(addprefix ./target/,$(PLUGIN).stack)
TARGET_VEC = $(addprefix ./target/,$(PLUGIN)-vec.bin)
ADSPFWBIN  = $(addprefix ./target/,$(PLUGIN)-fw.bin)
TARGETS = \
	$(TARGET_ELF) \
	$(TARGET_BIN) \
	$(TARGET_DBG) \
	$(TARGET_SIZ) \
	$(TARGET_SYM) \
	$(TARGET_LST) \
	$(TARGET_STK) \
	$(TARGET_VEC) \
	$(ADSPFWBIN)

LIBS = $(wildcard ./target/libs/*.a)
PREBUILTS = $(wildcard ./prebuilts/libs/*.a)
all : $(TARGETS)
LIBGCC_EH = ./prebuilts/libgcc_eh.a.local

WITH_DEBUG_LIBS := 0
ifeq ($(WITH_DEBUG_LIBS), 1)
	PREBUILTS += ./prebuilts/libs/debug/libadspdebug.a
endif

WITH_COMPRESSED_IMAGE := 0
include loader/loader.mk

$(TARGET_VEC): $(TARGET_ELF)
	@echo generating image: $@
	$(SIZE) $<
	$(OBJCOPY) --only-section=".text.vectortable" -O binary $< $@

ifeq ($(WITH_COMPRESSED_IMAGE), 0)
$(ADSPFWBIN): $(TARGET_VEC) $(TARGET_BIN)
	@echo generating image: $@
	cat $(TARGET_VEC) $(TARGET_BIN) > $@
else
$(ADSPFWBIN): $(TARGET_VEC) $(NV_PLATFORM_APE_LOADER_DIR)/loader.bin
	@echo generating image: $@
	cat $(TARGET_VEC) $(NV_PLATFORM_APE_LOADER_DIR)/loader.bin > $@
endif

$(TARGET_ELF): $(LIBS) $(PREBUILTS)
	$(NOECHO)$(LD) -L . --nmagic  -T ../ape-segments.ld --sort-common --allow-multiple-definition --whole-archive $(LIBS) $(PREBUILTS) --no-whole-archive $(LIBGCC) $(LIBSTDC) $(LIBGCC_EH) -o $@
#	$(OBJCOPY) --keep-global-symbol=_app_ex_$(PLUGIN) $@

$(TARGET_BIN): $(TARGET_ELF)
	$(SIZE) $<
	$(OBJCOPY) --remove-section=".text.vectortable" --remove-section=".mbox_shared_data" --remove-section=".debug_mem_logs" -O binary $< $@

$(TARGET_DBG): $(TARGET_ELF)
	$(OBJDUMP) -Mreg-names-raw -S $< | $(CPPFILT) > $@

$(TARGET_SIZ): $(TARGET_ELF)
	$(NM) -S --size-sort $< > $@

$(TARGET_SYM): $(TARGET_ELF)
	$(OBJDUMP) -t $< | $(CPPFILT) > $@

$(TARGET_LST): $(TARGET_ELF)
	$(OBJDUMP) -Mreg-names-raw -d $< | $(CPPFILT) > $@

$(TARGET_STK): $(TARGET_LST)
	$(NV_PLATFORM_APE_STACK_USAGE) < $< | sort -n -k 1 -r > $@

clean clobber:
	rm -rf $(TARGETS) $(NV_PLATFORM_APE_TARGET_LIBS)/* $(NV_PLATFORM_APE_LOADER_DIR) $(TARGET_BIN_XZ)
