OUTPUT_FORMAT("elf32-littlearm", "elf32-littlearm", "elf32-littlearm")
OUTPUT_ARCH(arm)

SECTIONS
{
	/* text/read-only data */
	.text : ALIGN(4096) {
		KEEP (*(.adsp_apps))
		. = ALIGN(4096);
		*(.text* .sram.text.glue_7* .gnu.linkonce.t.*)
	}

	.interp : { *(.interp) }
	.hash : { *(.hash) }
	.dynsym : { *(.dynsym) }
	.dynstr : { *(.dynstr) }
	.rel.text : { *(.rel.text) *(.rel.gnu.linkonce.t*) }
	.rela.text : { *(.rela.text) *(.rela.gnu.linkonce.t*) }
	.rel.data : { *(.rel.data) *(.rel.gnu.linkonce.d*) }
	.rela.data : { *(.rela.data) *(.rela.gnu.linkonce.d*) }
	.rel.rodata : { *(.rel.rodata) *(.rel.gnu.linkonce.r*) }
	.rela.rodata : { *(.rela.rodata) *(.rela.gnu.linkonce.r*) }
	.rel.got : { *(.rel.got) }
	.rela.got : { *(.rela.got) }
	.rel.ctors : { *(.rel.ctors) }
	.rela.ctors : { *(.rela.ctors) }
	.rel.dtors : { *(.rel.dtors) }
	.rela.dtors : { *(.rela.dtors) }
	.rel.init : { *(.rel.init) }
	.rela.init : { *(.rela.init) }
	.rel.fini : { *(.rel.fini) }
	.rela.fini : { *(.rela.fini) }
	.rel.bss : { *(.rel.bss) }
	.rela.bss : { *(.rela.bss) }
	.rel.plt : { *(.rel.plt) }
	.rela.plt : { *(.rela.plt) }
	.init : { *(.init) } =0x9090
	.plt : { *(.plt) }

	/* .ARM.exidx is sorted, so has to go in its own output section.  */
	__exidx_start = .;
	.ARM.exidx : { *(.ARM.exidx* .gnu.linkonce.armexidx.*) }
	__exidx_end = .;

	.rodata : ALIGN(4096) {
		*(.rodata .rodata.* .gnu.linkonce.r.*)
		. = ALIGN(4);
	}

	.data : ALIGN(4096) {
		*(.data .data.* .gnu.linkonce.d.*)
	}

	.dram_data : ALIGN(8) {
		KEEP (*(.dram_data))
	}

	.dram_shared : ALIGN(8) {
		KEEP (*(.dram_shared))
	}

	.dram_shared_wc : ALIGN(8) {
		KEEP (*(.dram_shared_wc))
	}

	.aram_data : ALIGN(8) {
		KEEP (*(.aram_data))
	}

	.aram_x_data : ALIGN(8) {
		KEEP (*(.aram_x_data))
	}

	.ctors : ALIGN(4096) {
		KEEP(*(.ctors .init_array))
	}
	.dtors : ALIGN(4096) {
		KEEP(*(.dtors .fini_array))
	}
	.got : { *(.got.plt) *(.got) }
	.dynamic : { *(.dynamic) }

	/* unintialized data (in same segment as writable data) */
	.bss : ALIGN(4096) {
		. = ALIGN(4096);
		*(.bss .bss.*)
		. = ALIGN(4096);
	}

	/* Strip unnecessary stuff */
	/DISCARD/ : { *(.comment .note .eh_frame) }
}
