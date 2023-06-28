OUTPUT_FORMAT("elf32-littlearm", "elf32-littlearm", "elf32-littlearm")
OUTPUT_ARCH(arm)

ENTRY(_loader_start)

PHDRS
{
	filehdr PT_LOAD;
	text PT_LOAD;
	data PT_LOAD;
	bss PT_NULL;
}

SECTIONS
{
	. = 2147483712;
	.text : {
		KEEP(*(.text.loader))
		KEEP(*(.text))
		KEEP(*(.rodata))
	} : text
	.data : {
		KEEP(*(.data))
		_fw_start = .;
		KEEP(*(.containers))
		_fw_end = .;
	} : data
	.bss : {
                __bss_start = .;
		KEEP(*(.bss.prebss.*))
		KEEP(*(.bss))
                __bss_end = .;
	} : bss
	.stack : {
		KEEP(*(.stack))
	} : stack
}

