OUTPUT_FORMAT("elf32-littlearm", "elf32-littlearm", "elf32-littlearm")
OUTPUT_ARCH(arm)

ENTRY(_start)

MEMORY
{
	VECTOR_TABLE (rx) : ORIGIN = 0x02993700, LENGTH = 0x40
	ADSP_OS_LAYOUT (rx) : ORIGIN = 0x80100000, LENGTH = 0x700000	 - 0x10000
}

SECTIONS
{
	. = 0x02993700;
	.text.vectortable : {
		KEEP(*(.text.vectortable))
	} > VECTOR_TABLE

	. = 0x80100000;
	_start_of_section_layout = .;
	/* text/read-only data */
	.text : {
		KEEP(*(.text.boot.vectab1))
		KEEP(*(.text.boot.vectab2))
		KEEP(*(.text.boot))
		*(.text* .sram.text.glue_7* .gnu.linkonce.t.*)
	} > ADSP_OS_LAYOUT

	.interp : { *(.interp) } > ADSP_OS_LAYOUT
	.hash : { *(.hash) } > ADSP_OS_LAYOUT
	.dynsym : { *(.dynsym) } > ADSP_OS_LAYOUT
	.dynstr : { *(.dynstr) } > ADSP_OS_LAYOUT
	.rel.text : { *(.rel.text) *(.rel.gnu.linkonce.t*) } > ADSP_OS_LAYOUT
	.rela.text : { *(.rela.text) *(.rela.gnu.linkonce.t*) } > ADSP_OS_LAYOUT
	.rel.data : { *(.rel.data) *(.rel.gnu.linkonce.d*) } > ADSP_OS_LAYOUT
	.rela.data : { *(.rela.data) *(.rela.gnu.linkonce.d*) } > ADSP_OS_LAYOUT
	.rel.rodata : { *(.rel.rodata) *(.rel.gnu.linkonce.r*) } > ADSP_OS_LAYOUT
	.rela.rodata : { *(.rela.rodata) *(.rela.gnu.linkonce.r*) } > ADSP_OS_LAYOUT
	.rel.got : { *(.rel.got) } > ADSP_OS_LAYOUT
	.rela.got : { *(.rela.got) } > ADSP_OS_LAYOUT
	.rel.ctors : { *(.rel.ctors) } > ADSP_OS_LAYOUT
	.rela.ctors : { *(.rela.ctors) } > ADSP_OS_LAYOUT
	.rel.dtors : { *(.rel.dtors) } > ADSP_OS_LAYOUT
	.rela.dtors : { *(.rela.dtors) } > ADSP_OS_LAYOUT
	.rel.init : { *(.rel.init) } > ADSP_OS_LAYOUT
	.rela.init : { *(.rela.init) } > ADSP_OS_LAYOUT
	.rel.fini : { *(.rel.fini) } > ADSP_OS_LAYOUT
	.rela.fini : { *(.rela.fini) } > ADSP_OS_LAYOUT
	.rel.bss : { *(.rel.bss) } > ADSP_OS_LAYOUT
	.rela.bss : { *(.rela.bss) } > ADSP_OS_LAYOUT
	.rel.plt : { *(.rel.plt) } > ADSP_OS_LAYOUT
	.rela.plt : { *(.rela.plt) } > ADSP_OS_LAYOUT
	.init : { *(.init) }  > ADSP_OS_LAYOUT =0x9090
	.plt : { *(.plt) } > ADSP_OS_LAYOUT

	/* .ARM.exidx is sorted, so has to go in its own output section.  */
	__exidx_start = .;
	.ARM.exidx : { *(.ARM.exidx* .gnu.linkonce.armexidx.*) } > ADSP_OS_LAYOUT
	__exidx_end = .;

	.rodata : ALIGN(4) {
		__rodata_start = .;
		*(.rodata .rodata.* .gnu.linkonce.r.*)
INCLUDE "arch/shared_rodata_sections.ld"
		. = ALIGN(4);
		__rodata_end = .;
	} > ADSP_OS_LAYOUT

	.data : ALIGN(4) {
		/* writable data  */
		__data_start_rom = .;
		/* in one segment binaries, the rom data address is on top of the ram data address */
		__data_start = .;
		*(.data .data.* .gnu.linkonce.d.*)
INCLUDE "arch/shared_data_sections.ld"
	} > ADSP_OS_LAYOUT

	.ctors : ALIGN(4) {
		__ctor_list = .;
		KEEP(*(.ctors .init_array))
		__ctor_end = .;
	} > ADSP_OS_LAYOUT
	.dtors : ALIGN(4) {
		__dtor_list = .;
		KEEP(*(.dtors .fini_array))
		__dtor_end = .;
	} > ADSP_OS_LAYOUT
	.got : { *(.got.plt) *(.got) } > ADSP_OS_LAYOUT
	.dynamic : { *(.dynamic) } > ADSP_OS_LAYOUT

	__data_end = .;

	.data_uncached : ALIGN(4096) {
		__data_uncached_start = .;
		KEEP(*(.data_uncached))
		__data_uncached_end = .;
	} > ADSP_OS_LAYOUT

	.dram_data : ALIGN(4096) {
		__dram_app_data_start = .;
		KEEP(*(.dram_data))
		__dram_app_data_end = .;
	} > ADSP_OS_LAYOUT

	.dram_shared_data : ALIGN(4096) {
		__dram_shared_app_data_start = .;
		KEEP(*(.dram_shared))
		__dram_shared_app_data_end = .;
	} > ADSP_OS_LAYOUT

	.dram_shared_wc_data : ALIGN(4096) {
		__dram_shared_wc_app_data_start = .;
		KEEP(*(.dram_shared_wc))
		__dram_shared_wc_app_data_end = .;
	} > ADSP_OS_LAYOUT

	.data_tests_uncached : ALIGN(4096) {
		__data_tests_uncached_start = .;
		KEEP(*(.data_tests_uncached))
		__data_tests_uncached_end = .;
	} > ADSP_OS_LAYOUT

	/* unintialized data (in same segment as writable data) */
	.bss : ALIGN(16318) {
		__bss_prebss_start = .;
		KEEP(*(.bss.prebss.*))
		__bss_prebss_end = .;

		. = ALIGN(4096);
		__bss_start = .;
		*(.bss .bss.*)
		*(.gnu.linkonce.b.*)
		*(COMMON)
		. = ALIGN(4);
		__bss_end = .;
	} > ADSP_OS_LAYOUT

	. = ALIGN(4096);
	_end = .;
	. = 0x80100000 + 0x700000	;
	_end_of_ram = .;

	_end_of_section_layout = .;

	/* Strip unnecessary stuff */
	/DISCARD/ : { *(.comment .note .eh_frame) }
}
