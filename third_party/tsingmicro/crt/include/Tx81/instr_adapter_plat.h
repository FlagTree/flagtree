#ifndef INSTR_ADAPTER_PLAT_H
#define INSTR_ADAPTER_PLAT_H

// You should define something, according to your device-type

// ==================== if you run in Tx8-simulator =====================================================

#include <stdint.h>


//#include "oplib_depend_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Data_Shape {
    uint16_t n;
    uint16_t h;
    uint16_t w;
    uint16_t c;
} Data_Shape;

typedef struct St_Elem_Shape {
    uint32_t elem_count;
    uint32_t unit_elem_count;
    uint32_t full_elem_count;
    uint32_t full_unit_elem_count;
} St_Elem_Shape;

typedef struct St_StrideIteration {
    uint32_t stride0;
    uint32_t iteration0;
    uint32_t stride1;
    uint32_t iteration1;
    uint32_t stride2;
    uint32_t iteration2;
} St_StrideIteration;

/*=================================C  class=================================*/
typedef struct TsmConv {
    void (*AddInput)(TsmNeInstr *instr, uint64_t X_addr, Data_Shape shape, Data_Format fmt);
    void (*AddWeight)(TsmNeInstr *instr, uint64_t W_addr, Data_Shape shape, Data_Format fmt);
    void (*AddBias)(TsmNeInstr *instr, uint8_t bias_en, uint64_t bias_addr);
    void (*AddOutput)(TsmNeInstr *instr, uint64_t Out_addr, Data_Shape shape, Data_Format fmt);
    void (*SetOpType)(TsmNeInstr *instr, uint8_t type);
    void (*SetNegativeAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t scale_addr); //- negative axis
    void (*SetPositiveAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t scale_addr); //+ positive axis
    void (*SetSparse)(TsmNeInstr *instr, uint8_t sparse_en, uint64_t sparse_addr);
    void (*SetPsum)(TsmNeInstr *instr, uint8_t psum_en, uint64_t psum_addr, Data_Format fmt);
    void (*SetPads)(TsmNeInstr *instr, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);
    void (*SetUnPads)(TsmNeInstr *instr, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);
    void (*SetKernelStrides)(TsmNeInstr *instr, uint32_t Kx, uint32_t Ky, uint32_t Sx, uint32_t Sy);
    void (*SetDilations)(TsmNeInstr *instr, uint32_t d0, uint32_t d1);
    void (*EnableRelu)(TsmNeInstr *instr);
    void (*EnableLeakyRelu)(TsmNeInstr *instr);
    void (*DisableRelu)(TsmNeInstr *instr);
    void (*DisableLeakyRelu)(TsmNeInstr *instr);
    void (*SetQuant)(TsmNeInstr *instr, uint8_t q0, uint8_t q1, uint8_t zp_pre, uint8_t zp_cur);
    /* data */
} TsmConv;

typedef struct TsmDepthwiseConv {
    void (*AddInput)(TsmNeInstr *instr, uint64_t X_addr, Data_Shape shape, Data_Format fmt);
    void (*AddWeight)(TsmNeInstr *instr, uint64_t W_addr, Data_Shape shape, Data_Format fmt);
    void (*AddBias)(TsmNeInstr *instr, uint8_t bias_en, uint64_t bias_addr);
    void (*AddOutput)(TsmNeInstr *instr, uint64_t Out_addr, Data_Shape shape, Data_Format fmt);
    void (*SetOpType)(TsmNeInstr *instr, uint8_t type);
    void (*SetNegativeAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t scale_addr); //- negative axis
    void (*SetPositiveAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t scale_addr); //+ positive axis
    void (*SetSparse)(TsmNeInstr *instr, uint8_t sparse_en, uint64_t sparse_addr);
    void (*SetPads)(TsmNeInstr *instr, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);
    void (*SetUnPads)(TsmNeInstr *instr, uint32_t top, uint32_t bottom, uint32_t left, uint32_t right);
    void (*SetKernelStrides)(TsmNeInstr *instr, uint32_t Kx, uint32_t Ky, uint32_t Sx, uint32_t Sy);
    void (*SetDilations)(TsmNeInstr *instr, uint32_t d0, uint32_t d1);
    void (*EnableRelu)(TsmNeInstr *instr);
    void (*EnableLeakyRelu)(TsmNeInstr *instr);
    void (*DisableRelu)(TsmNeInstr *instr);
    void (*DisableLeakyRelu)(TsmNeInstr *instr);
    void (*SetQuant)(TsmNeInstr *instr, uint8_t q0, uint8_t q1, uint8_t zp_pre, uint8_t zp_cur);
    /* data */
} TsmDepthwiseConv;
typedef struct TsmGemm {
    void (*AddInput)(TsmNeInstr *instr, uint64_t L_addr, uint64_t R_addr, Data_Format in_fmt);
    void (*ConfigMKN)(TsmNeInstr *instr, uint32_t M, uint32_t K, uint32_t N);
    void (*ConfigBatch)(TsmNeInstr *instr, uint32_t Left_batch, uint32_t Right_batch);
    void (*AddOutput)(TsmNeInstr *instr, uint64_t Out_addr, Data_Format Out_fmt);
    void (*SetPsum)(TsmNeInstr *instr, uint8_t psum_en, uint64_t psum_addr, Data_Format fmt);
    void (*SetTransflag)(TsmNeInstr *instr, uint8_t L_trans, uint8_t R_trans);
    void (*SetQuant)(TsmNeInstr *instr, uint8_t q0, uint8_t q1, uint8_t zp_left, uint8_t zp_right);
    void (*AddBias)(TsmNeInstr *instr, uint8_t bias_en, uint64_t addr);
    void (*SetNegativeAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t addr);
    void (*SetPositiveAxisScale)(TsmNeInstr *instr, uint8_t scale_en, uint64_t addr);
    void (*EnableRelu)(TsmNeInstr *instr);
    void (*EnableLeakyRelu)(TsmNeInstr *instr);
    void (*DisableRelu)(TsmNeInstr *instr);
    void (*DisableLeakyRelu)(TsmNeInstr *instr);

    /* data */
} TsmGemm;
typedef struct TsmRdma {
    void (*AddSrcDst)(TsmRdmaInstr *instr, uint64_t src, uint64_t dst, Data_Format fmt);
    void (*ConfigStrideIteration)(TsmRdmaInstr *instr, uint32_t elem_count, uint32_t stride0, uint32_t iteration0,
                                  uint32_t stride1, uint32_t iteration1, uint32_t stride2, uint32_t iteration2);
    void (*Rdma1d)(TsmRdmaInstr *instr, uint64_t src, uint64_t dst, uint32_t elem_count,
                   uint32_t format); //只有stride0,和iteration0,内层循环, 只复制一次
} TsmRdma;

typedef struct TsmWdma {
    void (*AddSrcDst)(TsmWdmaInstr *instr, uint64_t src, uint64_t dst, Data_Format fmt);
    void (*ConfigStrideIteration)(TsmWdmaInstr *instr, uint32_t elem_count, uint32_t stride0, uint32_t iteration0,
                                  uint32_t stride1, uint32_t iteration1, uint32_t stride2, uint32_t iteration2);
    void (*Wdma1d)(TsmWdmaInstr *instr, uint64_t src, uint64_t dst, uint32_t elem_count,
                   uint32_t format); //只有stride0,和iteration0,内层循环, 只复制一次
} TsmWdma;



/*=================================CGRA=================================*/
typedef struct TsmArith {
    void(*AbsVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void(*RecipVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void(*SquareVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void(*SqrtVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void(*RsqrtVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void(*NegVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                            Data_Format fmt);
    void (*MaxVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                            uint32_t elem_count, RND_MODE reserved, Data_Format fmt);
    void (*MaxVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                              uint32_t elem_count, RND_MODE reserved, Data_Format fmt);
    void (*MaxVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                              uint32_t elem_count, uint32_t unit_elem_count, RND_MODE reserved, Data_Format fmt);
    void (*MaxVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                                    uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                                    uint32_t full_unit_elem_num, RND_MODE reserved, Data_Format fmt);
    void(*MinVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                              uint32_t elem_count, RND_MODE reserved, Data_Format fmt);
     void(*MinVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                              uint32_t elem_count, RND_MODE reserved, Data_Format fmt);
    void(*MinVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                                uint32_t elem_count, uint32_t unit_elem_count, RND_MODE reserved, Data_Format fmt);
    void(*MinVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                                    uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                                    uint32_t full_unit_elem_num, RND_MODE reserved, Data_Format fmt);
    void (*AddVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  RND_MODE rnd_mode, Data_Format fmt);
    void (*AddVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                  uint32_t elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*AddVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                    uint32_t unit_elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*AddVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                        uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                        uint32_t full_unit_elem_num, RND_MODE rnd_mode, Data_Format fmt);
    void (*SubVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  RND_MODE rnd_mode, Data_Format fmt);
    void (*SubVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                  uint32_t elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*SubVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                    uint32_t unit_elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*SubVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                        uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                        uint32_t full_unit_elem_num, RND_MODE rnd_mode, Data_Format fmt);
    void (*MulVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  RND_MODE rnd_mode, Data_Format fmt);
    void (*MulVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                  uint32_t elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*MulVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  uint32_t unit_elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*MulVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, RND_MODE rnd_mode, Data_Format fmt);
    void (*DivVV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  RND_MODE rnd_mode, Data_Format fmt);
    void (*DivVS)(TsmArithInstr *instr, uint64_t src0_addr, uint32_t const_value, uint64_t dst_addr,
                  uint32_t elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*DivVuV)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count,
                  uint32_t unit_elem_count, RND_MODE rnd_mode, Data_Format fmt);
    void (*DivVuVLoop)(TsmArithInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, RND_MODE rnd_mode, Data_Format fmt);
} TsmArith;


typedef struct TsmRelation {
    void (*EqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*EqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*EqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*EqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*UnEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolUnEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*UnEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolUnEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*UnEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolUnEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*UnEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolUnEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*GreaterEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolGreaterEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*GreaterEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolGreaterEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*GreaterEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolGreaterEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*GreaterEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolGreaterEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*GreaterVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolGreaterVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*GreaterVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolGreaterVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*GreaterVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolGreaterVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*GreaterVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolGreaterVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*LessEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolLessEqualVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*LessEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolLessEqualVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*LessEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolLessEqualVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*LessEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolLessEqualVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*LessThenVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolLessThenVV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*LessThenVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*BoolLessThenVS)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t convst_value, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*LessThenVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*BoolLessThenVuV)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*LessThenVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*BoolLessThenVuVLoop)(TsmRelationInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
} TsmRelation;

typedef struct TsmLogic {
    void (*NotV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*AndVV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*OrVV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*XorVV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*AndVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*OrVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*XorVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count, Data_Format fmt);
    void (*AndVuVLoop)(TsmArithInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*OrVuVLoop)(TsmArithInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);
    void (*XorVuVLoop)(TsmArithInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr,
                      uint32_t elem_count, uint32_t unit_elem_count, uint32_t full_elem_num,
                      uint32_t full_unit_elem_num, Data_Format fmt);

    void (*BoolNotV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BoolAndV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BoolOrV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BoolXorV)(TsmLogicInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BoolAndVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count);
    void (*BoolOrVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count);
    void (*BoolXorVuV)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count, uint32_t unit_elem_count);
    void (*BoolAndVuVLoop)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count,
                           uint32_t unit_elem_count, uint32_t full_elem_num, uint32_t full_unit_elem_num);
    void (*BoolOrVuVLoop)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count,
                          uint32_t unit_elem_count, uint32_t full_elem_num, uint32_t full_unit_elem_num);
    void (*BoolXorVuVLoop)(TsmLogicInstr *instr, uint64_t src_addr, uint64_t unit_addr, uint64_t dst_addr, uint32_t elem_count,
                           uint32_t unit_elem_count, uint32_t full_elem_num, uint32_t full_unit_elem_num);
} TsmLogic;

typedef struct TsmTranscendental {
    void (*Log2)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Ln)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Pow2)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Exp)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Explp)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Sin)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Cos)(TsmArithInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
} TsmTranscendental;

typedef struct TsmActivation {
    void (*Tanh)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Sigmoid)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Relu)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Satrelu)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Leakyrelu)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*Softplus)(TsmActivationInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
} TsmActivation;

typedef struct TsmReduce {
    void (*ReduceSum)(TsmReduceInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t dim, Data_Shape shape,
                      Data_Format fmt);
    void (*ReduceAvg)(TsmReduceInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t dim, Data_Shape shape,
                      Data_Format fmt);
    void (*ReduceMax)(TsmReduceInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t dim, Data_Shape shape,
                      Data_Format fmt);
    void (*ReduceMin)(TsmReduceInstr *instr, uint64_t src_addr, uint64_t dst_addr, uint32_t dim, Data_Shape shape,
                      Data_Format fmt);
} TsmReduce;

typedef struct TsmPool {
    void (*MaxPool)(TsmPoolInstr *instr, uint64_t src0, Data_Shape src_shape, uint64_t dst, Data_Shape pad,
                    Data_Shape swr_shape, Data_Format fmt);
    void (*AvgPool)(TsmPoolInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr, Data_Shape pad,
                    Data_Shape swr_shape, Data_Format fmt);
    void (*SumPool)(TsmPoolInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr, Data_Shape pad,
                    Data_Shape swr_shape, Data_Format fmt);
    void (*MinPool)(TsmPoolInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr, Data_Shape pad,
                    Data_Shape swr_shape, Data_Format fmt);
    void (*IndexdMinPool)(TsmPoolInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_arg,
                          uint64_t dst_idx, Data_Shape pad, Data_Shape swr_shape, Data_Format fmt);
    void (*IndexdMaxPool)(TsmPoolInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_arg,
                          uint64_t dst_idx, Data_Shape pad, Data_Shape swr_shape, Data_Format fmt);
} TsmPool;

typedef struct TsmUnPool {
    void (*Unpool)(TsmUnPoolInstr *instr, uint64_t src0_addr, uint32_t index, uint64_t dst_addr, Data_Shape dst_shape, Data_Shape swr_shape, Data_Format fmt);
    void (*UnpoolAvg)(TsmUnPoolInstr *instr, uint64_t src0_addr, uint64_t dst_addr, Data_Shape dst_shape, Data_Shape swr_shape, Data_Format fmt);
    void (*UnpoolIdx)(TsmUnPoolInstr *instr, uint64_t src0_addr, uint32_t index, uint64_t dst_addr, Data_Shape dst_shape, Data_Shape swr_shape, Data_Format fmt);
} TsmUnPool;

typedef struct TsmMaskDataMove {
    void (*MaskMove)(TsmMaskDataMoveInstr *instr, uint64_t src0_addr, uint32_t mask, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*MaskGather)(TsmMaskDataMoveInstr *instr, uint64_t src0_addr, uint32_t index, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*MaskGather_bV)(TsmMaskDataMoveInstr *instr, uint64_t src0_addr, uint32_t bitindex, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
} TsmMaskDataMove;


typedef struct TsmConvert {
    void (*INT8_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint32_t zp, uint64_t dst_addr, uint32_t elem_count); // Data_Format fmt is INT8
    void (*INT8_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint32_t zp, uint64_t dst_addr, uint32_t elem_count);
    void (*INT8_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint32_t zp, uint64_t dst_addr, uint32_t elem_count);
    void (*INT8_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint32_t zp, uint64_t dst_addr, uint32_t elem_count);
    void (*INT16_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count); // INT16
    void (*INT16_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*INT16_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*INT16_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);

    void (*INT32_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*INT32_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*INT32_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*INT32_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);

    void (*BF16_INT8)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BF16_INT16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*BF16_INT32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*BF16_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BF16_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*BF16_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);

    void (*FP16_INT8)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP16_INT16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP16_INT32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP16_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode); // rnd_mode 0~4
    void (*FP16_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*FP16_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);

    void (*FP32_INT8)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP32_INT16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP32_INT32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP32_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode); // rnd_mode 0~4
    void (*FP32_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*FP32_TF32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);

    void (*TF32_INT8)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*TF32_INT16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*TF32_INT32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);
    void (*TF32_FP16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
    void (*TF32_BF16)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, RND_MODE rnd_mode);// rnd_mode 0~4
    void (*TF32_FP32)(TsmConvertInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count);
} TsmConvert;

typedef struct TsmPeripheral {
    void (*Count)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count,
                  Data_Format fmt, uint64_t *wb_data0, uint64_t *wb_data1);
    void (*Memset)(TsmDataMoveInstr *instr, uint64_t dst_addr, uint32_t value, uint32_t elem_count,
                  St_StrideIteration *si, Data_Format fmt); // si.stride is byte size. but ele_count is only element count
    void (*Bit2Fp)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t elem_count, Data_Format fmt);
    void (*ArgMax)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint32_t elem_count, Data_Format fmt);
    void (*ArgMin)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint32_t elem_count, Data_Format fmt);
    void (*Bilinear)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint64_t dst0_addr, Data_Shape src_shape,
                  Data_Shape dst_shape, int32_t scale_w, int32_t scale_h, Data_Format fmt);
    void (*Lut16)(TsmPeripheralInstr *instr, uint64_t src1_addr, uint64_t dst0_addr, uint64_t lut16_addr,
                  uint32_t src_elem_count, uint32_t lut_elem_count);
    void (*Lut32)(TsmPeripheralInstr *instr, uint64_t src1_addr, uint64_t dst0_addr, uint64_t lut32_addr,
                  uint32_t src_elem_count, uint32_t lut_elem_count);
    void (*RandGen)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint64_t src1_addr, uint64_t dst_addr,
                  uint64_t dst1_addr, uint64_t dst2_addr, uint32_t src_elem_num, Data_Format fmt);
    void (*Factorize)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint64_t dst1_addr, uint64_t dst2_addr,
                      uint32_t src_elem_num);
    void (*ElemMask)(TsmPeripheralInstr *instr, uint64_t src0_addr, uint32_t scale, uint64_t dst_addr, uint32_t src_elem_num, Data_Format fmt,
                     uint32_t prob, RND_MODE rnd_mode);
} TsmPeripheral;

typedef struct TsmDataMove {
    void (*Mirror)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                    Data_Shape dst_shape, Data_Format fmt);
    void (*Transpose)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*Rotate90)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                    Data_Shape dst_shape, Data_Format fmt);
    void (*Rotate180)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*Rotate270)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*Nchw2nhwc)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*Nhwc2nchw)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*Concat)(TsmMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape0, uint64_t src1_addr,
                Data_Shape src_shape1, uint64_t dst_addr, Data_Shape dst_shape, uint32_t dims, Data_Format fmt);
    void (*Pad)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                Data_Shape dst_shape, Data_Shape pad, Data_Format fmt);
    void (*Img2col)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                    Data_Shape dst_shape, uint64_t src_elem_num, uint64_t dst_elem_num, Data_Shape swr, Data_Shape pdr,
                    Data_Format fmt);
    void (*TensorNom)(TsmDataMoveInstr *instr, uint64_t src0_addr, Data_Shape src_shape, uint64_t dst_addr,
                      Data_Shape dst_shape, Data_Format fmt);
    void (*GatherScatter)(TsmDataMoveInstr *instr, uint64_t src0_addr, uint64_t dst_addr, uint32_t size,
                          St_StrideIteration *src_si, St_StrideIteration *dst_si);
} TsmDataMove;

TsmConv *TsmNewConv();
TsmDepthwiseConv *TsmNewDepthwiseConv();
TsmGemm *TsmNewGemm();
TsmRdma *TsmNewRdma();
TsmWdma *TsmNewWdma();
TsmArith *TsmNewArith();
TsmRelation *TsmNewRelation();
TsmLogic *TsmNewLogic();
TsmTranscendental *TsmNewTranscendental();
TsmActivation *TsmNewActivation();
TsmReduce *TsmNewReduce();
TsmPool *TsmNewPool();
TsmUnPool *TsmNewUnPool();
TsmMaskDataMove *TsmNewMaskDataMove();
TsmConvert *TsmNewConvert();
TsmPeripheral *TsmNewPeripheral();
TsmDataMove *TsmNewDataMove();

void TsmDeleteConv(TsmConv *obj);
void TsmDeleteDepthwiseConv(TsmDepthwiseConv *obj);
void TsmDeleteGemm(TsmGemm *obj);
void TsmDeleteRdma(TsmRdma *obj);
void TsmDeleteWdma(TsmWdma *obj);
void TsmDeleteArith(TsmArith *obj);
void TsmDeleteRelation(TsmRelation *obj);
void TsmDeleteLogic(TsmLogic *obj);
void TsmDeleteTranscendental(TsmTranscendental *obj);
void TsmDeleteActivation(TsmActivation *obj);
void TsmDeleteReduce(TsmReduce *obj);
void TsmDeletePool(TsmPool *obj);
void TsmDeleteUnPool(TsmUnPool *obj);
void TsmDeleteMaskDataMove(TsmMaskDataMove *obj);
void TsmDeleteConvert(TsmConvert *obj);
void TsmDeletePeripheral(TsmPeripheral *obj);
void TsmDeleteDataMove(TsmDataMove *obj);
/*=================================STREAM=================================*/
typedef struct TsmStream {
    uint32_t (*OnlineStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint32_t (*OfflineStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint32_t (*WaitStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint32_t (*ReqStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint32_t (*PushStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint32_t (*PopStream)(uint32_t core_id_this, uint32_t tile_id, uint32_t core_id, uint32_t channel_id,
        uint32_t remote, uint32_t stream_type, uint64_t stream_id, uint64_t stream_addr);
    uint8_t (*wait_finish)();
} TsmStream;
TsmStream *TsmNewStream();
void TsmDeleteStream(TsmStream *obj);
/*=================================CSR=====================================*/
uint8_t TsmWaitfinish();
uint8_t TsmGetCsrTaskstatus();
uint8_t TsmGetCsrIbcounter();
uint8_t TsmGetCsrTaskstatus_bywork(size_t workerid);
uint8_t TsmWaitfinish_bywork(size_t workerid);
/*=================================CSR END=================================*/
#ifdef __cplusplus
}
#endif

// ==================== if you will use Tx8-Oplib ======================================================
// #define LOG_PRINT(...)
// #define LOG_ERR(fmt, args...)
// #define TSM_FREE free
// #define TSM_MALLOC malloc
// extern void setreg(int index, uint64_t value);
// extern  uint64_t getreg(int index);

// ==================== if you run in SOC-freerots/zebu ================================================
// #include "rce_log.h"
// #include "csi_kernel.h"
// #include "rce_pal.h"
// #define LOG_PRINT(fmt, args...) vdk_printf(fmt, ##args)
// #define LOG_ERR(fmt, args...) vdk_printf(fmt, ##args)
// #define TSM_FREE(target) csi_kernel_free(2, target, NULL)
// #define TSM_MALLOC(size) csi_kernel_malloc(2, size, NULL)
// #define NCC_ADDR 0x01000000
// #define setreg(ADDR, VALUE)
//     do {
//         LOG_PRINT("(FREERT)setreg params: GR: index=0x%X, value=0x%lX(%lu).\n", ADDR, VALUE, VALUE);
//         *((volatile uint64_t *)(ADDR + NCC_ADDR)) = VALUE;
//     } while (0)

// ==================== if you run in kernel-rt(use tile-sim) ==========================================
// #define LOG_PRINT(fmt, args...) printf(fmt, ##args)
// #define LOG_ERR(fmt, args...) printf(fmt, ##args)
// #define TSM_FREE free
// #define TSM_MALLOC malloc
// #include "rce_pal_port.h"
// void setreg(int index, uint64_t value)
// {
//     LOG_PRINT("setreg param: GR:  index=0x%X, value=0x%lX(%lu).\n", index, value, value);
//     rce_tx_pal_setreg(index, value);
// }

// ====================if you run in kernel-rt(use riscv) ===============================================
// #define LOG_PRINT(...)
// #define LOG_ERR(fmt, args...)
// #define TSM_FREE free
// #define TSM_MALLOC malloc
// #define NCC_ADDR 0x01000000
// #define setreg(ADDR, VALUE)
//     do {
//         LOG_PRINT("(FREERT)setreg params: GR: index=0x%X, value=0x%lX(%lu).\n", ADDR, VALUE, VALUE);
//         *((volatile uint64_t *)(ADDR + NCC_ADDR)) = VALUE;
//     } while (0)

// ====================if you do not need Log ==========================================================
//#define LOG_PRINT(...)
//#define LOG_ERR(fmt, args...)
#endif
