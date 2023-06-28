-include ./../../../make/nvapedefs.mk

WITH_COMPRESSED_IMAGE := 1
MEMBASE := 0x80100000

ifeq ($(WITH_COMPRESSED_IMAGE), 1)
TARGET_BIN_XZ := $(TARGET_BIN).xz

$(TARGET_BIN_XZ): $(TARGET_BIN)
	$(NOECHO)xz -f --keep --check=none --lzma2=,dict=1MiB $<

$(NV_PLATFORM_APE_LOADER_DIR)/loader.S: ./loader/loader.S $(TARGET_BIN_XZ)
	@mkdir -p $(NV_PLATFORM_APE_LOADER_DIR)
	$(NOECHO)sed "s#%OUTBIN%#$(realpath $(TARGET_BIN))#;" $< > $@

$(NV_PLATFORM_APE_LOADER_DIR)/loader.o: $(NV_PLATFORM_APE_LOADER_DIR)/loader.S
	$(NOECHO)$(CC) -DMEMBASE=$(MEMBASE) -I$(NV_PLATFORM_APE_COMMON_INC) -c $< -o $@

$(NV_PLATFORM_APE_LOADER_DIR)/loader.elf: ./loader/loader.ld $(NV_PLATFORM_APE_LOADER_DIR)/loader.o ./loader/libloader.a
	$(NOECHO)$(LD) -T $< $(filter-out $<, $^) -o $@

$(NV_PLATFORM_APE_LOADER_DIR)/loader.bin: $(NV_PLATFORM_APE_LOADER_DIR)/loader.elf
	$(NOECHO)$(OBJCOPY) -O binary $< $@
endif
