/*
 * Copyright (C) 2017-2024 Alibaba Group Holding Limited
 */

/******************************************************************************
 * @file     gcc_csky.ld
 * @brief    csky linker file
 * @version  V1.0
 * @date     02. June 2017
 ******************************************************************************/
MEMORY
{
    mem0 (rwx) : ORIGIN = 0x00000000, LENGTH = (20*1024*1024)
}

REGION_ALIAS("r", mem0);
REGION_ALIAS("w", mem0);
REGION_ALIAS("x", mem0);

ENTRY(Reset_Handler)
SECTIONS
{

.text.startup 0x0:{
   . = ALIGN(0x8) ;
   KEEP(*startup.o(*.text.startup))
   *(.text.startup)
 } > x

 .text : {
  . = ALIGN(0x8) ;
  __ram_code_start__ = .;
  __stext = . ;
  KEEP(*startup.o(*.text))
  KEEP(*startup.o(*.vectors))
  KEEP(*vectors.o(*.text))
  KEEP(*(.text.entry))
  *(.text)
  *(.vectors)
  *(.text*)
  *(.text.*)
  *(.gnu.warning)
  *(.stub)
  *(.gnu.linkonce.t*)
  *(.glue_7t)
  *(.glue_7)
  *(.jcr)
  KEEP (*(.init))
  KEEP (*(.fini))
  . = ALIGN(0x8) ;
  PROVIDE(__ctbp = .);
  *(.call_table_data)
  *(.call_table_text)
  . = ALIGN(0x8) ;
  __etext = . ;
  __ram_code_end__ = .;
 } > x

 .gcc_except_table : ONLY_IF_RO {
  *(.gcc_except_table .gcc_except_table.*)
 } > x

 .rodata : {
  . = ALIGN(0x8) ;
  __srodata = .;
  *(.rdata)
  *(.rdata*)
  *(.rdata1)
  *(.rdata.*)
  *(.rodata)
  *(.rodata1)
  *(.rodata*)
  *(.rodata.*)
  *(.rodata.str1.4)
  *(.srodata*)
  . = ALIGN(0x8) ;

  __init_array_start = .;
  __ctors_start__ = .;
  KEEP (*(SORT(.init_array.*)))
  KEEP (*(.init_array))
  __init_array_end = .;
  __ctors_end__ = .;

  __fini_array_start = .;
  __dtors_start__ = .;
  KEEP (*(SORT(.fini_array.*)))
  KEEP (*(.fini_array))
  __fini_array_end = .;
  __dtors_end__ = .;

  __ctor_start__ = .;
  KEEP (*(SORT(.ctors.*)))
  KEEP (*(.ctors))
  __ctor_end__ = .;
  KEEP (*(SORT(.dtors.*)))
  KEEP (*(.dtors))
  __dtor_end__ = .;
  . = ALIGN(0x8) ;
/*****************************************/
  /* section information for finsh shell */
  . = ALIGN(0x8);
  __fsymtab_start = .;
  KEEP(*(FSymTab))
  __fsymtab_end = .;
  . = ALIGN(0x8);
  __vsymtab_start = .;
  KEEP(*(VSymTab))
  __vsymtab_end = .;
  . = ALIGN(0x8);

  /* section information for initial. */
  __rt_init_start = .;
  KEEP(*(SORT(.rti_fn*)))
  __rt_init_end = .;
  . = ALIGN(0x8) ;

  /* section information for at utest */
  __rt_utest_tc_tab_start = .;
  KEEP(*(UtestTcTab))
  __rt_utest_tc_tab_end = .;
  . = ALIGN(0x8);

  /* section information for at server */
  . = ALIGN(0x8);
  __rtatcmdtab_start = .;
  KEEP(*(RtAtCmdTab))
  __rtatcmdtab_end = .;
  . = ALIGN(0x8);

  /* section information for modules */
  . = ALIGN(0x8);
  __rtmsymtab_start = .;
  KEEP(*(RTMSymTab))
  __rtmsymtab_end = .;

  /* section information for uPRC */
  . = ALIGN(0x8);
  __uRPCSvcTab_start = .;
  KEEP(*(uRPCSvcTab))
  __uRPCSvcTab_end = .;

  /* section information for var export */
  . = ALIGN(0x8);
  __ve_table_start = .;
  KEEP(*(SORT(*.VarExpTab.*)))
  __ve_table_end = .;
/*****************************************/
/************** added drivers **************/
  _cli_region_begin = .;
  KEEP(*(CliRegion))
  . = ALIGN(0x8) ;
  _cli_region_end = .;

  __core_driver_start__ = .;
  KEEP(*(.core_driver_entry))
  . = ALIGN(0x8) ;
  __core_driver_end__ = .;

  __bus_driver_start__ = .;
  KEEP(*(*.bus_driver_entry))
  __bus_driver_end__ = .;

  __early_driver_start__ = .;
  KEEP(*(*.early_driver_entry))
  __early_driver_end__ = .;

  __vfs_driver_start__ = .;
  KEEP(*(*.vfs_driver_entry))
  __vfs_driver_end__ = .;

  __level0_driver_start__ = .;
  KEEP(*(*.level0_driver_entry))
  __level0_driver_end__ = .;

  __level1_driver_start__ = .;
  KEEP(*(*.level1_driver_entry))
  __level1_driver_end__ = .;

  __level2_driver_start__ = .;
  KEEP(*(*.level2_driver_entry))
  __level2_driver_end__ = .;

  __level3_driver_start__ = .;
  KEEP(*(*.level3_driver_entry))
  __level3_driver_end__ = .;

  __post_driver_start__ = .;
  KEEP(*(*.post_driver_entry))
  __post_driver_end__ = .;
/************** end of drivers *********/
  . = ALIGN(0x8) ;
  __erodata = .;
  __rodata_end__ = .;
 } > r

 .data : {
  . = ALIGN(0x8) ;
  __sdata = . ;
  __data_start__ = . ;
  data_start = . ;
  *(.got.plt)
  *(.got)
  *(.gnu.linkonce.r*)
  *(.data)
  *(.data*)
  *(.data1)
  *(.data.*)
  *(.gnu.linkonce.d*)
  *(.data1)
  *(.gcc_except_table)
  *(.gcc_except_table*)
  __start_init_call = .;
  *(.initcall.init)
  __stop_init_call = .;
  __start_cmd = .;
  *(.bootloaddata.cmd)
  . = ALIGN(0x8) ;
  __stop_cmd = .;
  __global_pointer$ = .;
  *(.sdata)
  *(.sdata.*)
  *(.sdata2.*)
  *(.gnu.linkonce.s.*)
  *(__libc_atexit)
  *(__libc_subinit)
  *(__libc_subfreeres)
  *(.note.ABI-tag)
  __edata = .;
  __data_end__ = .;
  . = ALIGN(0x8) ;
 } >  w AT> r

 .gcc_except_table : ONLY_IF_RW {
  *(.gcc_except_table .gcc_except_table.*)
  __edata = .;
  __data_end__ = .;
 } >  w AT> r

 .rela.dyn : {
      . = ALIGN(0x8) ;
     __rel_dyn_start = .;
     *(.rela*)
     __rel_dyn_end = .;
 }
 .dynsym : {
      . = ALIGN(0x8) ;
     __dyn_sym_start = .;
     *(.dynsym)
     __dyn_sym_end = .;
 }
 .bss : {
  . = ALIGN(0x8) ;
  __sbss = ALIGN(0x8) ;
  __bss_start__ = . ;
  *(.dynsbss)
  *(.sbss)
  *(.sbss.*)
  *(.scommon)
  *(.dynbss)
  *(.bss)
  *(.bss.*)
  *(COMMON)
  . = ALIGN(0x8) ;
  __ebss = . ;
  __bss_end__ = .;
  __end = . ;
  end = . ;
 } >  w
 ._user_heap (NOLOAD): {
  . = ALIGN(0x8) ;
  *(.stack*)
  . = ALIGN(0x8) ;
  __heap_start = ABSOLUTE(.);
  . = ORIGIN(w) + LENGTH(w);
  __heap_end = ABSOLUTE(.);

 } > w
}
