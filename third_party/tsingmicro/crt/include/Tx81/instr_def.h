#ifndef _RCE_INSTR_DEF_H_
#define _RCE_INSTR_DEF_H_
#include <stdint.h>

#define UN_USED 0

// CT
#define GR_CT_CONTROL_ADDR 0x0000
#define GR_CT_SRC0_ADDR 0x0008 * 2
#define GR_CT_SRC1_ADDR 0x0010 * 2
#define GR_CT_DST0_ADDR 0x0018 * 2
#define GR_CT_DST1_ADDR 0x0020 * 2
#define GR_CT_DST2_ADDR 0x0028 * 2
#define GR_CT_DIMS_ADDR 0x0030 * 2
#define GR_CT_SRC0_TFR_ADDR 0x0038 * 2
#define GR_CT_DST_TFR_ADDR 0x0040 * 2
#define GR_CT_PDR_ADDR 0x0048 * 2
#define GR_CT_SWR_ADDR 0x0050 * 2
#define GR_CT_ELEM_COUNT_ADDR 0x0058 * 2
#define GR_CT_UNIT_ELEM_COUNT_ADDR 0x0060 * 2
#define GR_CT_INT8_SCALE_VAL0_ADDR 0x0068 * 2
#define GR_CT_INT8_SCALE_VECTOR_ADDR 0x0068 * 2
#define GR_CT_INT8_SCALE_VAL1_ADDR 0x0070 * 2
#define GR_CT_INT8_QUANT_ADDR 0x0078 * 2
#define GR_CT_INT8_BN_ZP_ADDR 0x0080 * 2
#define GR_CT_FULL_ELEM_COUNT_ADDR 0x0088 * 2
#define GR_CT_FULL_UNIT_ELEM_COUNT_ADDR 0x0090 * 2
#define GR_CT_WB_DATA0_ADDR 0x0098 * 2
#define GR_CT_WB_DATA1_ADDR 0x00A0 * 2
#define GR_CT_SRC0_END_ADDR 0x00A8 * 2
#define GR_CT_SRC1_END_ADDR 0x00B0 * 2
#define GR_CT_DST0_END_ADDR 0x00B8 * 2
#define GR_CT_DST1_END_ADDR 0x00C0 * 2
#define GR_CT_DST2_END_ADDR 0x00C8 * 2

// NE
#define GR_NE_CONTROL_ADDR 0x0100 * 2
#define GR_NE_SRC_A_ADDR 0x0108 * 2
#define GR_NE_SRC_W_ADDR 0x0110 * 2
#define GR_NE_PSUM_ADDR 0x0118 * 2
#define GR_NE_BIAS_ADDR 0x0120 * 2
#define GR_NE_SCALE_P_ADDR 0x0128 * 2
#define GR_NE_SCALE_N_ADDR 0x0130 * 2
#define GR_NE_OUT_ADDR 0x0138 * 2
#define GR_NE_SRC0_TFR_ADDR 0x0140 * 2
#define GR_NE_SRC1_OUT_TFR_ADDR 0x0148 * 2
#define GR_NE_PDR_ADDR 0x0150 * 2
#define GR_NE_UNPDR_ADDR 0x0158 * 2
#define GR_NE_SWR_ADDR 0x0160 * 2
#define GR_NE_DILATION_ADDR 0x0168 * 2
#define GR_NE_GEMM_LB_ADDR 0x0170 * 2
#define GR_NE_GEMM_RB_ADDR 0x0178 * 2
#define GR_NE_GEMM_N_ADDR 0x0180 * 2
#define GR_NE_GEMM_M_ADDR 0x0188 * 2
#define GR_NE_GEMM_K_ADDR 0x0190 * 2
#define GR_NE_GEMM_L_TRS_ADDR 0x0198 * 2
#define GR_NE_GEMM_R_TRS_ADDR 0x01A0 * 2
#define GR_NE_QUANT_ADDR 0x01A8 * 2
#define GR_NE_SPARSE_INDEX_ADDR 0x01B0 * 2
#define GR_NE_SRCA_END 0x370
#define GR_NE_SRCW_END 0x380
#define GR_NE_PSUM_END 0x390
#define GR_NE_BIAS_END 0x3A0
#define GR_NE_SCALE_P_END 0x3B0
#define GR_NE_SCALE_N_END 0x3C0
#define GR_NE_OUT_END 0x3D0
#define GR_NE_SPARSE_INDEX_END 0x3E0

// LSU RDMA
#define GR_RD_CONTROL_ADDR 0x400
#define GR_RD_SRC_ADDR 0x410
#define GR_RD_DST_ADDR 0x420
#define GR_RD_STRIDE_ITERA0_ADDR 0x430
#define GR_RD_STRIDE_ITERA1_ADDR 0x440
#define GR_RD_STRIDE_ITERA2_ADDR 0x450
#define GR_RD_ELEM_COUNT_ADDR 0x460
#define GR_RD_FORMAT_ADDR 0x470
#define GR_RD_SRC_END 0x480
#define GR_RD_DST_END 0x490
// LSU WDMA
#define GR_WD_CONTROL_ADDR 0x4A0
#define GR_WD_SRC_ADDR 0x4B0
#define GR_WD_DST_ADDR 0x4C0
#define GR_WD_STRIDE_ITERA0_ADDR 0x4D0
#define GR_WD_STRIDE_ITERA1_ADDR 0x4E0
#define GR_WD_STRIDE_ITERA2_ADDR 0x4F0
#define GR_WD_ELEM_COUNT_ADDR 0x500
#define GR_WD_FORMAT_ADDR 0x510
#define GR_WD_SRC_END 0x520
#define GR_WD_DST_END 0x530

// TMDA
#define GR_TD_CONTROL_ADDR 0x02A0 * 2
#define GR_TD_SRC0_ADDR 0x02A8 * 2
#define GR_TD_SRC1_ADDR 0x02B0 * 2
#define GR_TD_DST_ADDR 0x02B8 * 2
#define GR_TD_DIMS_ADDR 0x02C0 * 2
#define GR_TD_SRC0_TFR_ADDR 0x02C8 * 2
#define GR_TD_DST_TFR_ADDR 0x02D0 * 2
#define GR_TD_PDR_ADDR 0x02D8 * 2
#define GR_TD_SWR_ADDR 0x02E0 * 2
#define GR_TD_ELEM_COUNT_ADDR 0x02E8 * 2
#define GR_TD_SRC_STRIDE_ITERA0_ADDR 0x02F0 * 2
#define GR_TD_SRC_STRIDE_ITERA1_ADDR 0x02F8 * 2
#define GR_TD_SRC_STRIDE_ITERA2_ADDR 0x0300 * 2
#define GR_TD_DST_STRIDE_ITERA0_ADDR 0x0308 * 2
#define GR_TD_DST_STRIDE_ITERA1_ADDR 0x0310 * 2
#define GR_TD_DST_STRIDE_ITERA2_ADDR 0x0318 * 2
#define GR_TD_SRC0_END 0x640
#define GR_TD_SRC1_END 0x650
#define GR_TD_DST_END 0x660
// SCALAR
#define GR_SCALAR_CONTROL_ADDR 0x6A0
#define GR_SCALAR_SRC_ADDR 0x6B0
#define GR_SCALAR_DST_ADDR 0x6C0
// CSR
#define GR_CSR_CONTROL_ADDR 0x740
#define GR_CSR_EXCEPTION_ADDR 0x750
#define GR_CSR_PRIORITY_ADDR 0x760
#define GR_CSR_EXCEPTION_MASK_ADDR 0x770
#define GR_CSR_SERIAL_MODE_ADDR 0x780

// CSR end
// DTE start
#define GR_DTE_SRC_ADDR_LO 0x0
#define GR_DTE_SRC_ADDR_HI 0x4
#define GR_DTE_DST_ADDR_LO_0 0x8
#define GR_DTE_DST_ADDR_HI_0 0xC
#define GR_DTE_USER_ID_0 0x10
#define GR_DTE_MODE 0x14
#define GR_DTE_LENGTH 0x18
#define GR_DTE_DEST_NUM 0x1C
#define GR_DTE_STRIDE0 0x20
#define GR_DTE_ITERATION0 0x24
#define GR_DTE_STRIDE1 0x28
#define GR_DTE_ITERATION1 0x2C
#define GR_DTE_STRIDE2 0x30
#define GR_DTE_ITERATION2 0x34
#define GR_DTE_CMD_VALID 0x38
#define GR_DTE_DMA_STATUS 0x40
#define GR_DTE_DST_ADDR_LO_1 0x50
#define GR_DTE_DST_ADDR_HI_1 0x54
#define GR_DTE_DST_ADDR_LO_2 0x58
#define GR_DTE_DST_ADDR_HI_2 0x5C
#define GR_DTE_DST_ADDR_LO_3 0x60
#define GR_DTE_DST_ADDR_HI_3 0x64
#define GR_DTE_DST_ADDR_LO_4 0x68
#define GR_DTE_DST_ADDR_HI_4 0x6C
#define GR_DTE_DST_ADDR_LO_5 0x70
#define GR_DTE_DST_ADDR_HI_5 0x74
#define GR_DTE_DST_ADDR_LO_6 0x78
#define GR_DTE_DST_ADDR_HI_6 0x7C
#define GR_DTE_DST_ADDR_LO_7 0x80
#define GR_DTE_DST_ADDR_HI_7 0x84
#define GR_DTE_DST_ADDR_LO_8 0x88
#define GR_DTE_DST_ADDR_HI_8 0x8C
#define GR_DTE_DST_ADDR_LO_9 0x90
#define GR_DTE_DST_ADDR_HI_9 0x94
#define GR_DTE_DST_ADDR_LO_10 0x98
#define GR_DTE_DST_ADDR_HI_10 0x9C
#define GR_DTE_DST_ADDR_LO_11 0xA0
#define GR_DTE_DST_ADDR_HI_11 0xA4
#define GR_DTE_DST_ADDR_LO_12 0xA8
#define GR_DTE_DST_ADDR_HI_12 0xAC
#define GR_DTE_DST_ADDR_LO_13 0xB0
#define GR_DTE_DST_ADDR_HI_13 0xB4
#define GR_DTE_DST_ADDR_LO_14 0xB8
#define GR_DTE_DST_ADDR_HI_14 0xBC
#define GR_DTE_DST_ADDR_LO_15 0xC0
#define GR_DTE_DST_ADDR_HI_15 0xC4
#define GR_DTE_DST_ADDR_LO_16 0xC8
#define GR_DTE_DST_ADDR_HI_16 0xCC
#define GR_DTE_DST_ADDR_LO_17 0xD0
#define GR_DTE_DST_ADDR_HI_17 0xD4
#define GR_DTE_DST_ADDR_LO_18 0xD8
#define GR_DTE_DST_ADDR_HI_18 0xD4
#define GR_DTE_DST_ADDR_LO_19 0xE0
#define GR_DTE_DST_ADDR_HI_19 0xE4
#define GR_DTE_DST_ADDR_LO_20 0xE8
#define GR_DTE_DST_ADDR_HI_20 0xEC
#define GR_DTE_DST_ADDR_LO_21 0xF0
#define GR_DTE_DST_ADDR_HI_21 0xF4
#define GR_DTE_DST_ADDR_LO_22 0xF8
#define GR_DTE_DST_ADDR_HI_22 0xFC
#define GR_DTE_DST_ADDR_LO_23 0x100
#define GR_DTE_DST_ADDR_HI_23 0x104
#define GR_DTE_DST_ADDR_LO_24 0x108
#define GR_DTE_DST_ADDR_HI_24 0x10C
#define GR_DTE_DST_ADDR_LO_25 0x110
#define GR_DTE_DST_ADDR_HI_25 0x114
#define GR_DTE_DST_ADDR_LO_26 0x118
#define GR_DTE_DST_ADDR_HI_26 0x11C
#define GR_DTE_DST_ADDR_LO_27 0x120
#define GR_DTE_DST_ADDR_HI_27 0x124
#define GR_DTE_DST_ADDR_LO_28 0x128
#define GR_DTE_DST_ADDR_HI_28 0x12C
#define GR_DTE_DST_ADDR_LO_29 0x130
#define GR_DTE_DST_ADDR_HI_29 0x134
#define GR_DTE_DST_ADDR_LO_30 0x138
#define GR_DTE_DST_ADDR_HI_30 0x13C
#define GR_DTE_DST_ADDR_LO_31 0x140
#define GR_DTE_DST_ADDR_HI_31 0x144

#define GR_DTE_USER_ID_1 0x148
#define GR_DTE_USER_ID_2 0x14C
#define GR_DTE_USER_ID_3 0x150
#define GR_DTE_USER_ID_4 0x154
#define GR_DTE_USER_ID_5 0x158
#define GR_DTE_USER_ID_6 0x15C
#define GR_DTE_USER_ID_7 0x160
#define GR_DTE_USER_ID_8 0x164
#define GR_DTE_USER_ID_9 0x168
#define GR_DTE_USER_ID_10 0x16C
#define GR_DTE_USER_ID_11 0x170
#define GR_DTE_USER_ID_12 0x174
#define GR_DTE_USER_ID_13 0x178
#define GR_DTE_USER_ID_14 0x17C
#define GR_DTE_USER_ID_15 0x180
#define GR_DTE_USER_ID_16 0x184
#define GR_DTE_USER_ID_17 0x188
#define GR_DTE_USER_ID_18 0x18C
#define GR_DTE_USER_ID_19 0x190
#define GR_DTE_USER_ID_20 0x194
#define GR_DTE_USER_ID_21 0x198
#define GR_DTE_USER_ID_22 0x19C
#define GR_DTE_USER_ID_23 0x1A0
#define GR_DTE_USER_ID_24 0x1A4
#define GR_DTE_USER_ID_25 0x1A8
#define GR_DTE_USER_ID_26 0x1AC
#define GR_DTE_USER_ID_27 0x1B0
#define GR_DTE_USER_ID_28 0x1B4
#define GR_DTE_USER_ID_29 0x1B8
#define GR_DTE_USER_ID_30 0x1BC
#define GR_DTE_USER_ID_31 0x1C0

#define GR_DTE_MAX_AXI_NUM 0x1D0
#define GR_DTE_MEM_BURSTLEN 0x1D4
#define GR_DTE_MEM_BACKPRESSURE 0x1D8
#define GR_DTE_MEM_READ_TURBO 0x1DC
// DTE end

// SCONFIG begin
#define GR_SCONFIG_GPR0 0x600

// SCONFIG end

// NCC PMU begin
#define GR_PMU_EN 0x0
#define GR_PMU_CLR 0x4
#define GR_PMU_STATISTICS_WINDOW 0x8
#define GR_PMU_CT_INST_NUMS 0x10
#define GR_PMU_NE_INST_NUMS 0x14
#define GR_PMU_RDMA_INST_NUMS 0x18
#define GR_PMU_WDMA_INST_NUMS 0x1C
#define GR_PMU_TDMA_INST_NUMS 0x20
#define GR_PMU_SCALAR_INST_NUMS 0x24
#define GR_PMU_CT_BLOCKING_TIME 0x28
#define GR_PMU_NE_BLOCKING_TIME 0x2C
#define GR_PMU_RDMA_BLOCKING_TIME 0x30
#define GR_PMU_WDMA_BLOCKING_TIME 0x34
#define GR_PMU_TDMA_BLOCKING_TIME 0x38
#define GR_PMU_SCALAR_BLOCKING_TIME 0x3c

#define GR_PMU_FU_EXE_TIME 0x13c
#define GR_PMU_CT_EXE_TIME 0x144
#define GR_PMU_NE_EXE_TIME 0x14c
#define GR_PMU_RDMA_EXE_TIME 0x154
#define GR_PMU_WDMA_EXE_TIME 0x15c
#define GR_PMU_TDMA_EXE_TIME 0x164
#define GR_PMU_SCALAR_EXE_TIME 0x16c
// NCC PMU end

// DTE PMU begin
#define DTE_PMU_EN 0x800
#define DTE_PMU_CLR 0x804

#define DTE_PMU_CH0_L_EXE_TIME 0x858
#define DTE_PMU_CH0_H_EXE_TIME 0x85C
#define DTE_PMU_CH1_L_EXE_TIME 0x860
#define DTE_PMU_CH1_H_EXE_TIME 0x864
// DTE PMU end

typedef enum OP_INSTR_TYPE {
    I_CGRA,
    I_NEUR,
    I_RDMA,
    I_WDMA,
    I_TDMA,
    I_SCALAR,
    I_DTE,
    I_CSR,
} OP_INSTR_TYPE;
// instr_type = I_CGRA | I_WORKER1
typedef enum OP_INSTR_WORKER {
    I_WORKER0 = 0x0000,
    I_WORKER1 = 0x0100,
    I_WORKER2 = 0x0200,
} OP_INSTR_WORKER;

typedef enum RND_MODE {
    RND_NEAREST_EVEN,
    RND_ZERO,
    RND_POS_INF,
    RND_NEG_INF,
    RND_STOCHASTIC
} RND_MODE;


typedef struct Ncc_CT_GR_Ctl_Regs {
    uint8_t cmd_valid;   // self clear
    uint8_t rnd_mode;    // 0 :round to nearest even , 1 :round to zero, 2 :round to positive infinity, 3 :round to
                         // negative infinity, 4 :stochastic round
    uint8_t src0_format; // 当CGRATensor_PeriOp_V_V_bit2fp指令，此字段用作dst_format
    uint8_t opcode;      // 详见CGRATensor指令OPcode.v
} Ncc_CT_GR_Ctl_Regs;

typedef struct Ncc_CT_GR_Param_Regs {
    uint32_t src0; // spm地址
    uint32_t src1;
    uint32_t dst0;
    uint32_t dst1;
    uint32_t dst2;                 // spm地址
    uint64_t src0_tfr;             // nhwc
    uint64_t dst_tfr;              // nhwc
    uint64_t pdr;                  // TOP BOTTOM,LEFT,RIGHT(分别是上下左右pad的行/列数)
    uint64_t swr;                  // kernel的 Kx(x方向的大小),Ky,Sx(x方向的步进),Sy
    uint64_t elem_count;           // vector运算的元素个数
    uint64_t unit_elem_count;      // vector运算中的短向量的元素个数(最大为64)
    uint64_t int8_scale_val0;      // 双线性插值x方向缩放系数(input_w/output_w)
    uint64_t int8_scale_val1;      // 双线性插值y方向缩放系数(input_h/output_h)
    uint64_t int8_quant;           // abandon
    uint32_t int8_bn_bias;         // abandon
    uint32_t full_elem_count;      // 若干个src_elem_num之和
    uint32_t full_unit_elem_count; // 若干个src_uint_elem_num之和
    uint64_t wb_data0;             // The pointer of Return value. [32] DATA_VALID, [31:0] data,
                                   // 函数只有一个返回值时，返回数据写在此寄存器
    uint64_t wb_data1;             // The pointer of Return value. [32] DATA_VALID, [31:0] data,
                       // 函数有两个返回值时，第二个返回数据写在此寄存器，当只有一个返回值时，此寄存器无效
    uint32_t src0_end; // spm地址(src0结束地址), xxx_end = src/dst + 对应操作数在spm中存储范围
    uint32_t src1_end;
    uint32_t dst0_end;
    uint32_t dst1_end;
    uint32_t dst2_end;
    uint8_t dims; // 000:C 001:W 010:H 011:N 100:HW 101:HWC
} Ncc_CT_GR_Param_Regs;

typedef struct CT_Param {
    uint32_t inter_type;
    Ncc_CT_GR_Ctl_Regs ctrl;
    Ncc_CT_GR_Param_Regs param;
} CT_Param;

#define TsmArithInstr CT_Param
#define TsmPoolInstr CT_Param
#define TsmMoveInstr CT_Param
#define TsmUnPoolInstr CT_Param
#define TsmMaskDataMoveInstr CT_Param
#define TsmConvertInstr CT_Param
#define TsmPeripheralInstr CT_Param
#define TsmRelationInstr CT_Param
#define TsmLogicInstr CT_Param
#define TsmTranscendentalInstr CT_Param
#define TsmActivationInstr CT_Param
#define TsmReduceInstr CT_Param

typedef struct Ncc_NE_GR_Ctl_Regs {
    uint8_t sparse_en;
    uint8_t cmd_valid;
    uint8_t inpsum_format;
    uint8_t output_format;
    uint8_t input_format;
    uint8_t inpsum_en;
    uint8_t lrelu_en; // either relu or lrelu
    uint8_t relu_en;  // relu_en/lrelu_en/bias_en/scale_en 同时为0时,输出是psum
    uint8_t scale_en;
    uint8_t bias_en;
    uint8_t dilation_conv; // valid as conv backwardconv
    uint8_t type;          // 0:conv 1:depthwise conv 2:backward conv 3:gemm
} Ncc_NE_GR_Ctl_Regs;

typedef struct Ncc_NE_GR_Param_Regs {
    uint32_t src_a;   // spm地址(激活/左矩阵)
    uint32_t src_w;   // spm地址(权重/右矩阵)
    uint32_t psum;    // spm地址(输入psum)
    uint32_t bias;    // spm地址(bias)
    uint32_t scale_p; // spm地址(正轴scale)
    uint32_t scale_n; // spm地址(负轴scale)
    uint32_t out;     // spm地址(输出psum)
    uint64_t tfr_0;   // src0 nhwc, [15:0]tensor batch/h/w(范围1~4096);tensor通道数(范围1~16384)
    uint64_t tfr_1;   // conv: out nhwc, 同上tfr_0
    uint64_t pdr;     // pad [15:0]top bottom left right, 分别是上下左右pad的行/列数(范围0~1023)
    uint64_t unpdr;   // unpad [15:0]top bottom left right
    uint64_t swr;     // [15:0]Kx(范围1~255) Ky(范围1~255) Sx(范围1~1023) Sy(范围1~1023)
    uint64_t dilation; // [15:0]空洞卷积的x方向大小(范围1-1023),  [15:0]空洞卷积的y方向大小(范围1-1023)

    uint16_t gemm_lb;   // [15:0]左矩阵batch(范围：1~4096)
    uint16_t gemm_rb;   // [15:0]左矩阵batch(范围：1~4096)
    uint16_t gemm_n;    // 矩阵运算的矩阵大小参数
    uint16_t gemm_m;    // mk*kn---->mn
    uint16_t gemm_k;    // (范围：1~16384)
    uint8_t gemm_l_trs; // 左矩阵转置
    uint8_t gemm_r_trs; // 右矩阵转置
    /*
       Quant formula----A_int8:Left input, B_int8: Right input
            Left input 8bit to 9bit: A_int9 = A_int8 - ZP_A_int8
            Left input 8bit to 9bit: A_int9 = A_int8 - ZP_A_int8
            do conv                : O_int32 = Sum_{A_int9 * B_int9}
            do scale               : O_int16 = Clip_int16(O_int32 >> q1)
            do scale               : O_int9 = Clip_int9((O_int16 * S_int16) >> q2)
            out 9bit to 8bit       : O_int8 = O_int9 + ZP_O_int8
    */
    uint8_t quant_zp_cur;   // 输出零点(0-255).   [39:32]
    uint8_t quant_reserved; //        (0-255). [31:24] conv:unused    gemm:right_zp
    uint8_t quant_zp_pre;   // 输入零点(0-255), [23:16] conv:act_zp    gemm:left_zp(范围：0-255)
    uint8_t quant_q1;       // q1, (范围：0-31),[15:8]
    uint8_t quant_q0;       // q2, (范围：0-31),[7:0]

    uint32_t sparse_index; // spm地址(稀疏化索引)
    uint32_t srca_end;     // xxx_end = src/dst + 对应操作数在spm中存储范围
    uint32_t srcw_end;
    uint32_t psum_end;
    uint32_t bias_end;
    uint32_t scale_p_end;
    uint32_t scale_n_end;
    uint32_t out_end;
    uint32_t sparse_end;
} Ncc_NE_GR_Param_Regs;

typedef struct TsmNeInstr {
    uint32_t inter_type;
    Ncc_NE_GR_Ctl_Regs ctrl;
    Ncc_NE_GR_Param_Regs param;
} TsmNeInstr;

// RDMA / WDMA
typedef struct Ncc_DMA_GR_Ctl_Regs {
    uint8_t cmd_valid;
} Ncc_DMA_GR_Ctl_Regs;

typedef struct Ncc_DMA_GR_Param_Regs {
    uint64_t dst; // ddr地址
    uint64_t src; // spm地址
    /*
        for(i = 0; i < itera2; i++)
            for(j = 0; j < itera1; j++)
                for(k = 0; k < itera0; k++)
                    for(l = 0; l < elem_count; l++)
                        dst[l + elem_coun * k + elem_coun * src_itera0 * j + elem_coun * src_itera0 * src_itera1 * i] =
       \ src[l + k * src_stride0 + j * src_stride1 + i * src_stride2];
    */
    uint32_t stride0;    //地址步长
    uint32_t iteration0; // 数据块个数
    uint32_t stride1;
    uint32_t iteration1;
    uint32_t stride2;
    uint32_t iteration2;
    uint32_t elem_count; // 最里面维度单次搬运的元素个数
    uint8_t format;      // 数据类型
    uint64_t src_end;    // src_end = src + ddr中数据存储长度
    uint64_t dst_end;    // dst_end = dst + spm中数据存储长度
} Ncc_DMA_GR_Param_Regs;

typedef struct DMA_Param {
    uint32_t inter_type;
    Ncc_DMA_GR_Ctl_Regs ctrl;
    Ncc_DMA_GR_Param_Regs param;
} DMA_Param;

#define TsmRdmaInstr DMA_Param
#define TsmWdmaInstr DMA_Param

typedef struct Ncc_TDMA_GR_Ctl_Regs {
    uint8_t cmd_valid;   // [12]
    uint8_t src0_format; // [11:8]
    uint8_t opcode;      //[7:0]
} Ncc_TDMA_GR_Ctl_Regs;

typedef struct Ncc_TDMA_GR_Param_Regs {
    uint32_t src0;
    uint32_t src1;
    uint32_t dst;
    uint64_t src0_tfr;   // nhwc  c:15~0
    uint64_t dst_tfr;    // nhwc
    uint64_t pdr;        // top bottom left right
    uint64_t swr;        // kx ky sx sy
    uint32_t elem_count; // vector操作的元素个数. memset、gatherscatter指令中代表byte number
    /*
        for(i=0;i<src_itera2;i++)
            for(j=0;j<src_itera1;j++)
                for(k=0;k<src_itera0;k++)
                    for(l=0;l<size;l++)
                        tmp[l+size*k+size*src_itera0*j+size*src_itera0*src_itera1*i]=src[l+k*src_stride0+j*src_stride1+i*src_stride2];

        for(i=0;i<dst_itera2;i++)
            for(j=0;j<dst_itera1;j++)
                for(k=0;k<dst_itera0;k++)
                    for(l=0;l<size;l++)
                        dst[l+k*dst_stride0+j*dst_stride1+i*dst_stride2]=tmp[l+size*k+size*dst_itera0*j+size*dst_itera0*dst_itera1*i];
    */
    uint32_t src_stride0;
    uint32_t src_iteration0;
    uint32_t src_stride1;
    uint32_t src_iteration1;
    uint32_t src_stride2;
    uint32_t src_iteration2;
    uint32_t dst_stride0;    // 地址步长
    uint32_t dst_iteration0; // 数据块个数
    uint32_t dst_stride1;
    uint32_t dst_iteration1;
    uint32_t dst_stride2;
    uint32_t dst_iteration2;
    uint32_t src0_end; // xxx_end = src/dst + 对应操作数在spm中存储范围
    uint32_t src1_end;
    uint32_t dst_end;
    uint8_t dims; //(3b) 000:C 001:W 010:H 011:N 100:HW 101:HWC
} Ncc_TDMA_GR_Param_Regs;

typedef struct TD_Param {
    uint32_t inter_type;
    Ncc_TDMA_GR_Ctl_Regs ctrl;
    Ncc_TDMA_GR_Param_Regs param;
} TD_Param;
#define TsmDataMoveInstr TD_Param
/*
    recip	8'b0000_0000
    sqrt	8'b0000_0001
    sin	    8'b0000_0010
    cos	    8'b0000_0011
    log2	8'b0000_0100
    pow2	8'b0000_0101
*/
typedef struct Ncc_SCALAR_GR_Ctl_Regs {
    uint8_t cmd_valid;
    uint8_t format;
    uint8_t opcode;
} Ncc_SCALAR_GR_Ctl_Regs;

typedef struct Ncc_SCALAR_GR_Param_Regs {
    uint32_t srcs; // 立即数
    uint32_t dst;  // 立即数，直接写回RF中
} Ncc_SCALAR_GR_Param_Regs;

typedef struct SC_Param {
    Ncc_SCALAR_GR_Ctl_Regs ctrl;
    Ncc_SCALAR_GR_Param_Regs param;
} SC_Param;

// csr Read or  write single;
typedef struct Ncc_CSR_GR_IB_STATUS_RO {
    uint8_t task_done;  // [8] 1：task执行结束 0：task 正在执行
    uint8_t in_counter; // [7:0] 指令buffer剩余指令数目
} Ncc_CSR_GR_IB_STATUS_RO;

typedef struct Ncc_CSR_GR_EXCEPTION_RO {
    uint8_t tdma_exception; // [47:40]
                            // Exception[0]：非法指令编码（opcode）
                            // Exception[1]：指令读/写地址超过3M范围
    uint8_t wdma_exception; // [39:32]
                            // Exception[3]: 非法指令（elem_count 等于0）
                            // Exception[2]：Fifo Overrun
                            // Exception[1]:  Axi Write Response != 0
                            // Exception[0]:  Axi Read Response != 0
    uint8_t rdma_exception; // [31:24]
                            // Exception[3]: 非法指令（elem_count 等于0）
                            // Exception[2]：Fifo Overrun
                            // Exception[1]:  Axi Write Response != 0
                            // Exception[0]:  Axi Read Response != 0
    uint8_t ne_exception;   // [23:16]
    uint8_t ct_exception;   // [15:8]
                            // Exception[0]：非法指令编码（opcode）
                            // Exception[1]：指令读/写地址超过3M范围
                            // Exception[2]：浮点运算结果出现无穷值（infinity）
                            // Exception[3]：浮点运算输入出现非数（NAN）
                          // Exception[4]：浮点数在不支持submormal的情况下，计算结果出现subnormal的数值。
                          // Exception[5]：浮点数计算结果超出最大表示范围
                          // Exception[6]：整数或者浮点数结果经过四舍五入后数值不精确出现误差
                          // Exception[7]：数据类型转换IP结果整数超出表示范围
    uint8_t scalar_exception; // [7:0]
                              // Exception[0]：浮点运算结果出现无穷值（infinity）
                              // Exception[1]：浮点运算输入出现非数（NAN）
                              // Exception[2]：浮点数在不支持submormal的情况下，计算结果出现subnormal的数值。
                              // Exception[3]：浮点数计算结果超出最大表示范围
                              // Exception[4]：整数或者浮点数结果经过四舍五入后数值不精确出现误差
                              // Exception[5]：数据类型转换IP结果整数超出表示范围
} Ncc_CSR_GR_EXCEPTION_RO;

typedef struct Ncc_CSR_GR_PRIORITY_RW {
    uint8_t priority; // [7:0] 前worker的优先级
} Ncc_CSR_GR_PRIORITY_RW;

typedef struct Ncc_CSR_GR_EXCEPTION_MASK {
    uint8_t exception_clear;         // [49] 清中断寄存器(self-clear,无需清零)
    uint8_t exception_update_enable; // [48] 1：保存最后一条异常 0：保存第一条异常，后续异常忽略
    uint64_t exception_mask; // [47:0] 48'hffff_ffff_ffff 异常使能 1：中断源被屏蔽 0：中断源未被屏蔽
} Ncc_CSR_GR_EXCEPTION_MASK;

typedef struct Ncc_CSR_GR_SERIAL_MODE {
    uint8_t serial_mode; // [0] 1：串行模式，所有指令进一个queue，不做指令间并行
                         // 0：并行模式，指令按照指令类型进独立的queue，检测依赖乱序发射 (测试默认为1，正式版本修改为0)
} Ncc_CSR_GR_SERIAL_MODE;
// csr end

typedef struct Ncc_DTE_GR_Param_Regs {
    uint8_t channel;      // value: 0~3
    uint8_t block;        // value: 0~1
    uint64_t dst[32];     // idx: 0~31, [0:31] low bit, [32:39] high bit
    uint16_t user_id[32]; // idx: 0~31, [0:15]
    uint64_t src;         // [31~0]: src_addr_lo, [32~63]: src_addr_hi
    uint32_t
        mode; // [1:0] mode:0->gather/scatter, unicast, 1-> scatter, broadcast, 3-> shuffle(3D gather). [8:8] sg_flag:
              // 0->scatter, 1->gather, [16:16] dim_flag: only unicast mode(=0), 0->1D transport, 1->2D transport
    uint32_t length;     // count data bytes
    uint8_t dest_num;    // if mode[0:0] is 0, then it's value is 1; otherwise it's value is between 1 and 31.
    uint32_t stride0;    // if mode[0:0] is 0, then stride can be setted, unit is byte.
    uint32_t iteration0; // 0: means 1 section; 1: means 2 sectons, and so on.
    uint32_t stride1;
    uint32_t iteration1;
    uint32_t stride2;
    uint32_t iteration2;
    uint16_t max_axi_num; // [7:0] axi_write_outstanding, [15:8] aix_read_outstanding
    uint8_t cmd_valid;    // 1: activate dma, 0: no action.
    // uint8_t dma_status; // [0:0] 0->unfinished, 1->finished. [8:8] 0/1, record the error of AXI bus or other DMA
    // transmission.
    uint16_t
        mem_burstlen; // [7:0] mem_burst_len_write, default value: 0x10; [15:8] mem_burst_len_read, default value: 0x10
    uint8_t mem_backpressure; // 0x1
    uint8_t mem_read_turbo;   // [1:0], 0~2, default value: 0, only block0 valid.
} Ncc_DTE_GR_Param_Regs;

typedef enum OP_FUNC_CGRA {
    // Arithmetic Operators
    OP_FUNC_CGRATensor_ArithOp_V_V_abs = 0,
    OP_FUNC_CGRATensor_ArithOp_V_V_recip = 1,
    OP_FUNC_CGRATensor_ArithOp_V_V_square = 2,
    OP_FUNC_CGRATensor_ArithOp_V_V_sqrt = 3,
    OP_FUNC_CGRATensor_ArithOp_V_V_rsqrt = 4,
    OP_FUNC_CGRATensor_ArithOp_V_V_neg = 5,
    OP_FUNC_CGRATensor_ArithOp_V_VV_max = 6,
    OP_FUNC_CGRATensor_ArithOp_V_VS_max = 7,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_max = 8,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_max_loop = 9,
    OP_FUNC_CGRATensor_ArithOp_V_VV_min = 10,
    OP_FUNC_CGRATensor_ArithOp_V_VS_min = 11,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_min = 12,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_min_loop = 13,
    OP_FUNC_CGRATensor_ArithOp_V_VV_add = 14,
    OP_FUNC_CGRATensor_ArithOp_V_VS_add = 15,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_add = 16,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_add_loop = 17,
    OP_FUNC_CGRATensor_ArithOp_V_VV_sub = 18,
    OP_FUNC_CGRATensor_ArithOp_V_VS_sub = 19,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_sub = 20,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_sub_loop = 21,
    OP_FUNC_CGRATensor_ArithOp_V_VV_mul = 22,
    OP_FUNC_CGRATensor_ArithOp_V_VS_mul = 23,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_mul = 24,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_mul_loop = 25,
    OP_FUNC_CGRATensor_ArithOp_V_VV_div = 26,
    OP_FUNC_CGRATensor_ArithOp_V_VS_div = 27,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_div = 28,
    OP_FUNC_CGRATensor_ArithOp_V_VuV_div_loop = 29,

    // Relational Operators
    OP_FUNC_CGRATensor_RelaOp_V_VV_eq = 30,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_eq = 31,
    OP_FUNC_CGRATensor_RelaOp_V_VS_eq = 32,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_eq = 33,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_eq = 34,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_eq_loop = 35,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_eq = 36,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_eq_loop = 37,

    OP_FUNC_CGRATensor_RelaOp_V_VV_ne = 38,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_ne = 39,
    OP_FUNC_CGRATensor_RelaOp_V_VS_ne = 40,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_ne = 41,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_ne = 42,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_ne_loop = 43,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_ne = 44,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_ne_loop = 45,

    OP_FUNC_CGRATensor_RelaOp_V_VV_ge = 46,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_ge = 47,
    OP_FUNC_CGRATensor_RelaOp_V_VS_ge = 48,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_ge = 49,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_ge = 50,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_ge_loop = 51,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_ge = 52,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_ge_loop = 53,

    OP_FUNC_CGRATensor_RelaOp_V_VV_gt = 54,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_gt = 55,
    OP_FUNC_CGRATensor_RelaOp_V_VS_gt = 56,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_gt = 57,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_gt = 58,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_gt_loop = 59,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_gt = 60,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_gt_loop = 61,

    OP_FUNC_CGRATensor_RelaOp_V_VV_le = 62,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_le = 63,
    OP_FUNC_CGRATensor_RelaOp_V_VS_le = 64,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_le = 65,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_le = 66,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_le_loop = 67,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_le = 68,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_le_loop = 69,

    OP_FUNC_CGRATensor_RelaOp_V_VV_lt = 70,
    OP_FUNC_CGRATensor_RelaOp_bV_VV_lt = 71,
    OP_FUNC_CGRATensor_RelaOp_V_VS_lt = 72,
    OP_FUNC_CGRATensor_RelaOp_bV_VS_lt = 73,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_lt = 74,
    OP_FUNC_CGRATensor_RelaOp_V_VuV_lt_loop = 75,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_lt = 76,
    OP_FUNC_CGRATensor_RelaOp_bV_VuV_lt_loop = 77,

    OP_FUNC_CGRATensor_LogicOp_V_V_not = 78,
    OP_FUNC_CGRATensor_LogicOp_V_VV_and = 79,
    OP_FUNC_CGRATensor_LogicOp_V_VV_or = 80,
    OP_FUNC_CGRATensor_LogicOp_V_VV_xor = 81,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_and = 82,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_or = 83,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_xor = 84,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_and_loop = 85,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_or_loop = 86,
    OP_FUNC_CGRATensor_LogicOp_V_VuV_xor_loop = 87,

    OP_FUNC_CGRATensor_LogicOp_bV_bV_not = 88,
    OP_FUNC_CGRATensor_LogicOp_bV_bVbV_and = 89,
    OP_FUNC_CGRATensor_LogicOp_bV_bVbV_or = 90,
    OP_FUNC_CGRATensor_LogicOp_bV_bVbV_xor = 91,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_and = 92,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_or = 93,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_xor = 94,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_and_loop = 95,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_or_loop = 96,
    OP_FUNC_CGRATensor_LogicOp_bV_bVubV_xor_loop = 97,

    // Transcendental Operator
    OP_FUNC_CGRATensor_TransOp_V_V_log2 = 98,
    OP_FUNC_CGRATensor_TransOp_V_V_ln = 99,
    OP_FUNC_CGRATensor_TransOp_V_V_pow2 = 100,
    OP_FUNC_CGRATensor_TransOp_V_V_exp = 101,
    OP_FUNC_CGRATensor_TransOp_V_V_exp_lp = 102,
    OP_FUNC_CGRATensor_TransOp_V_V_sin = 103,
    OP_FUNC_CGRATensor_TransOp_V_V_cos = 104,

    // Activation Operator
    OP_FUNC_CGRATensor_ActOp_V_V_tanh = 105,
    OP_FUNC_CGRATensor_ActOp_V_V_sigmoid = 106,
    OP_FUNC_CGRATensor_ActOp_V_V_relu = 107,
    OP_FUNC_CGRATensor_ActOp_V_V_satrelu = 108,
    OP_FUNC_CGRATensor_ActOp_V_V_leakyrelu = 109,
    OP_FUNC_CGRATensor_ActOp_V_V_softplus = 110,

    // Reduce Operator
    OP_FUNC_CGRATensor_ReduceOp_T_T_sum = 111,
    OP_FUNC_CGRATensor_ReduceOp_T_T_avg = 112,
    OP_FUNC_CGRATensor_ReduceOp_T_T_max = 113,
    OP_FUNC_CGRATensor_ReduceOp_T_T_min = 114,

    // Pool Operator
    OP_FUNC_CGRATensor_PoolOp_T_T_avg = 115,
    OP_FUNC_CGRATensor_PoolOp_T_T_sum = 116,
    OP_FUNC_CGRATensor_PoolOp_T_T_max = 117,
    OP_FUNC_CGRATensor_PoolOp_T_T_indexedmax = 118,
    OP_FUNC_CGRATensor_PoolOp_T_T_min = 119,
    OP_FUNC_CGRATensor_PoolOp_T_T_indexedmin = 120,

    // DataMove
    OP_FUNC_CGRATensor_DataMoveOp_T_T_unpool = 121,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_unpool_avg = 122,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_maskunpool = 123,
    // reshape
    OP_FUNC_CGRATensor_DataMoveOp_T_T_mirror = 124,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_transpose = 125,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_rotate90 = 126,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_rotate180 = 127,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_rotate270 = 128,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_nchw2nhwc = 129,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_nhwc2nchw = 130,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_concat = 131,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_pad = 132,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_channelnorm = 133,
    // datamove
    OP_FUNC_CGRATensor_DataMoveOp_V_V_maskmove = 134,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_gatherscatter = 135,
    OP_FUNC_CGRATensor_DataMoveOp_V_V_maskgather = 136,
    OP_FUNC_CGRATensor_DataMoveOp_V_bV_maskgather = 137,
    OP_FUNC_CGRATensor_DataMoveOp_T_T_img2col = 138,

    // Conver Operator
    OP_FUNC_CGRATensor_ConvertOp_V_V_int8_fp16 = 139,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int8_bf16 = 140,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int8_fp32 = 141,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int8_tf32 = 142,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int16_fp16 = 143,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int16_bf16 = 144,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int16_fp32 = 145,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int16_tf32 = 146,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int32_fp16 = 147,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int32_bf16 = 148,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int32_fp32 = 149,
    OP_FUNC_CGRATensor_ConvertOp_V_V_int32_tf32 = 150,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_int8 = 151,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_int16 = 152,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_int32 = 153,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_fp16 = 154,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_fp32 = 155,
    OP_FUNC_CGRATensor_ConvertOp_V_V_bf16_tf32 = 156,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_int8 = 157,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_int16 = 158,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_int32 = 159,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_bf16 = 160,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_fp32 = 161,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp16_tf32 = 162,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_int8 = 163,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_int16 = 164,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_int32 = 165,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_fp16 = 166,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_bf16 = 167,
    OP_FUNC_CGRATensor_ConvertOp_V_V_fp32_tf32 = 168,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_int8 = 169,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_int16 = 170,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_int32 = 171,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_fp16 = 172,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_bf16 = 173,
    OP_FUNC_CGRATensor_ConvertOp_V_V_tf32_fp32 = 174,

    // Peripheral Operator
    OP_FUNC_CGRATensor_PeriOp_S_V_count = 175,
    OP_FUNC_CGRATensor_PeriOp_S_bV_bitcount = 176,
    OP_FUNC_CGRATensor_PeriOp_V_V_argmax = 177,
    OP_FUNC_CGRATensor_PeriOp_V_V_argmin = 178,
    OP_FUNC_CGRATensor_PeriOp_T_memset = 179,
    OP_FUNC_CGRATensor_PeriOp_V_V_fp32_factorize = 180,
    OP_FUNC_CGRATensor_PeriOp_V_V_bit2fp = 181,
    OP_FUNC_CGRATensor_PeriOp_T_T_bilinear = 182,
    OP_FUNC_CGRATensor_PeriOp_V_V_lut16 = 183,
    OP_FUNC_CGRATensor_PeriOp_V_V_lut32 = 184,
    OP_FUNC_CGRATensor_PeriOp_V_rand_gen = 185,
    OP_FUNC_CGRATensor_PeriOp_V_V_elem_mask = 186,
} OP_FUNC_CGRA;

typedef enum CGRA_INSTR_TYPE {
    CGRA_INSTR_TYPE0,
    CGRA_INSTR_TYPE1,
    CGRA_INSTR_TYPE2,
    CGRA_INSTR_TYPE3,
} CGRA_INSTR_TYPE;

typedef struct Op_fu_head {
    uint8_t fu;
    uint8_t opcode;
} Op_fu_head;

typedef struct FU_gemm_head {
    uint8_t fu;
    uint8_t gemm;
} FU_gemm_head;

typedef struct opfunc_cgra_info {
    char name[64];  // CGRATensor_ArithOp_V_V_abs
    int32_t opcode; // 8'b0000_0000
    int32_t type;   // CGRA_Tensor_type0
} opfunc_cgra_info;

// Neural
typedef enum Data_Format {
    Fmt_INT8,
    Fmt_INT16,
    Fmt_FP16,
    Fmt_BF16,
    Fmt_INT32,
    Fmt_FP32,
    Fmt_TF32,
    Fmt_BOOL,    // 1/8 BYTE
    Fmt_UINT8,
    Fmt_UINT16,
    Fmt_UINT32,
    Fmt_INT64,
    Fmt_UINT64,
    Fmt_UNUSED,
} Data_Format;

typedef enum Tensor_Fmt {
    T_GemmM = 0, /*M K*/
    T_ConvA = 1, /*H W C*/
    T_ConvW = 2, /*Kx Ky F C*/
    T_Vec = 3,
    T_ConvNA = 4,
    T_ConvNW = 5,
} Tensor_Fmt;

/*
    张量做SumReduce操作，支持以下维度：
    C方向规约，结果为HW(C=1)，dim=0
    W方向规约，结果为H(W=1)C，dim=1
    H方向规约，结果为(H=1)WC，dim=2
    HW方向规约，结果为(H=1)(W=1)C，dim=4
*/
typedef enum Reduce_Dim {
    Reduce_C = 0,
    Reduce_W = 1,
    Reduce_H = 2,
    Reduce_HW = 4,
} Reduce_Dim;

typedef struct NCC_CSR {
    uint64_t ib_status; //[7:0]IB_COUNTER: 指令buffer剩余指令数目, [8]TASK_DONE, 1：task执行结束, 0：task 正在执行,
                        //[63:9]Reserved
    uint64_t exception;      //[7:0]SCALAR_EXCEPTION, [15:8]CT_EXCEPTION, [23:16]NE_EXCEPTION, [31:24]RDMA_EXCEPTION,
                             //[39:32]WDMA_EXCEPTION, [47:40]TDMA_EXCEPTION, [63:48]Reserved
    uint64_t priority;       //[7:0]PRIORITY,当前worker的优先级, [63:8]Reserved
    uint64_t exception_mask; //[47:0]EXCEPTION_MASK, [48]EXCEPTION_UPDATE_ENABLE, [49]EXCEPTION_CLEAR, [63:49]Reserved
    uint64_t serial_mode;    //[0]SERIAL_MODE, 1：串行模式，0：并行模式, [63:1]Reserved
} NCC_CSR;

typedef struct EXCEP_SERI {
    uint64_t exception_mask; //[47:0]EXCEPTION_MASK, [48]EXCEPTION_UPDATE_ENABLE, [49]EXCEPTION_CLEAR, [63:49]Reserved
    uint64_t serial_mode;    //[0]SERIAL_MODE, 1：串行模式，0：并行模式, [63:1]Reserved
} EXCEP_SERI;
#endif
