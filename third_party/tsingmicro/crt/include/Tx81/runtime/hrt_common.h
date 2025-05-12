/*
 * Copyright (C) 2024 Tsing Micro Intelligent Technology Co.,Ltd. All rights
 * reserved.
 *
 * This file is the property of Tsing Micro Intelligent Technology Co.,Ltd. This
 * file may only be distributed to: (i) a Tsing Micro party having a legitimate
 * business need for the information contained herein, or (ii) a non-Tsing Micro
 * party having a legitimate business need for the information contained herein.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef __HOST_RUNTIME_COM_H__
#define __HOST_RUNTIME_COM_H__

#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

#ifndef MAX_SHAPE_DIM
#define MAX_SHAPE_DIM 6
#endif

#ifndef MAX_MODEL_NUM
#define MAX_MODEL_NUM 32
#endif

typedef uint64_t TsmDevicePtr;
typedef uint64_t TsmHostPtr;

#define CHIP_MAX_NUM 32
#define TILE_MAX_NUM 16
#define CACHE_ALIGN_4k 4096

typedef void *(*THREAD_PROC_FUNC)(void *);

enum TSM_RETCODE {
  RET_SUCCESS,
  RET_ERROR,
  RET_PARAM1_ERROR,
  RET_PARAM2_ERROR,
  RET_PARAM3_ERROR,
  RET_DEVICE_OFFLINE,
  RET_DEVICE_NOMEM,
  RET_DEVICE_IN_IDLE,
  RET_DEVICE_IN_ATTACH,
  RET_DEVICE_ATTACH_SUCCESS,
  RET_DEVICE_ATTACH_READY,
  RET_DEVICE_LOSE_CONNECT,
  RET_ENV_CLEAN_UP,
};

typedef enum HostLogLevel {
  LOG_DEBUG,
  LOG_INFO,
  LOG_WARNING,
  LOG_ERROR,
  LOG_FATAL,
  LOG_MAX
} HostLogLevel;

typedef enum TsmModuleType {
  TSM_RUNTIME,
  TSM_XLA,     // 前端
  TSM_TXNN,    // 推理引擎
  TSM_ENGTEST, // 板端测试套件
  TSM_HOSTSIM, // 模拟器测试套件
  TSM_CMODEL,  // 模拟器API
  TSM_RT_TEST, // runtime组件测试套件
} TsmModuleType;

typedef enum TsmProfAction { TSM_PROF_START, TSM_PROF_STOP } TsmProfAction;
constexpr uint16_t PROF_TYPE_NCC = 0x1;
constexpr uint16_t PROF_TYPE_SPM = 0x2;
constexpr uint16_t PROF_TYPE_DTE = 0x4;

typedef enum DTYPE {
  FMT_INT8,
  FMT_INT16,
  FMT_FP16,
  FMT_BF16,
  FMT_INT32,
  FMT_FP32,
  FMT_TF32,
  FMT_BOOL, // 1/8 BYTE
  FMT_UINT8,
  FMT_UINT16,
  FMT_UINT32,
  FMT_INT64,
  FMT_UINT64,
  FMT_UNUSED,
} DTYPE;

uint8_t hrt_get_dtype_size(DTYPE dtype);

enum DynDataType {
  PKT_FINAL_TYPE = 0,
  CFG_PMU_TYPE,
  KCORE_CFG_TYPE,
  EXPORT_SPM_TYPE,
  DISABLE_CALC_TYPE,
  PROF_CFG_TYPE,
  DYNLIB_LOAD,
  DYNLIB_RUN,
  DYNLIB_UNLOAD,
  MEMCPY_D2D,
  P2P_SEND,
  P2P_RECV,
  DATA_TYPE_MAX,
};

typedef struct DynTLV_Terminate {
  uint32_t type; // DynDataType
  uint32_t len;
  uint64_t is_final;
} DynTLV_Terminate;

typedef struct DynTLV {
  uint32_t type; // DynDataType
  uint32_t len;
} DynTLV;

typedef struct Cfg_Pmu_Info {
  uint32_t tile_bitmap[16];
  uint32_t mac_use_rate;
  uint32_t chip_id;
  uint32_t cycles;
  uint64_t in_ddr;
  uint64_t param_ddr;
  uint64_t out_ddr;
  uint32_t reserved;
} Cfg_Pmu_Info;

typedef struct DynTLV_Cfgpmu {
  uint32_t type; // DynDataType
  uint32_t len;
  Cfg_Pmu_Info cfg_pmu;
} DynTLV_Cfgpmu;

typedef struct DynTLV_KcoreCfg {
  uint32_t type;
  uint32_t len;
  uint64_t snap_addr[TILE_MAX_NUM];
  uint64_t console_addr[TILE_MAX_NUM];
  uint64_t spm_dump_addr[TILE_MAX_NUM];
  uint64_t spm_dump_size;
  uint32_t log_level;
  uint32_t enable_monitor;
} DynTLV_KcoreCfg;

typedef struct DynTLV_KcoreCalc {
  uint32_t type;
  uint32_t len;
  uint32_t disable_kcore_calc;
} DynTLV_KcoreCalc;

typedef struct DynTLV_ProfCfg {
  uint32_t type;
  uint32_t len;
  uint64_t addrs[TILE_MAX_NUM];
  uint32_t size;
  uint16_t enable;
  uint16_t prof_type;
} DynTLV_ProfCfg;

// #define TILE_NUM 16
typedef struct DynModule {
  char module_name[128];
  char module_symbol[128]; // typedef void (*entry_func_t)(voicd *):
  uint32_t module_size[TILE_MAX_NUM];
  uint64_t module_addr[TILE_MAX_NUM]; // dev地址
} DynModule;

typedef struct DynMods {
  uint16_t module_num;
  struct DynModule modules[0];
} DynMods; // host共用结构,传过来这个首地址

typedef struct DynTLV_DynMods {
  uint32_t type; // DynDataType
  uint32_t len;
  uint64_t ext_addr;
  uint64_t dyn_mods_addr; // 指向DynMods
} DynTLV_DynMods;

typedef struct TileDteCfg {
  uint16_t status;             // 该tile是否参与搬运工作
  uint16_t remote_tile_id;     // 对端tile_id
  uint32_t element_count;      // 单次搬运cache_line大小，默认4k
  uint32_t stride;             // 步长
  uint32_t left_element_count; // 搬完cache_line后，剩余的搬运的长度
  uint64_t iteration;          // 搬运cache_line的次数
  uint64_t src_addr;           // 搬运cache_line的源地址 - 物理
  uint64_t dst_addr;           // 搬运cache_line的目的地址 - 物理
  uint64_t left_src_addr;      // 搬运余数的源地址 - 物理
  uint64_t left_dst_addr;      // 搬运余数的目的地址 - 物理
} TileDteCfg;
typedef struct DynTLV_DteCfg {
  uint32_t type;
  uint32_t len;
  TileDteCfg tile_dte_cfg[TILE_MAX_NUM];
  uint64_t barrier_addr;
  uint32_t row_card_num;
  uint32_t reserved;
} DynTLV_DteCfg;

enum Tensor_Type {
  INPUT_DATA,
  OUTPUT_DATA,
  PARAM_DATA,
  CHACHE_DATA,
  DEV_DDR_DATA,
};

typedef struct tensor_info {
  int32_t inplace;
  uint32_t dim;
  uint32_t dtype;
  uint32_t layout;
  uint32_t shape[MAX_SHAPE_DIM];
} tensor_info_t;

typedef struct Json_common_info_t {
  uint32_t input_num;
  uint32_t output_num;
  uint32_t param_num;
  uint32_t tile_num;

  std::string case_name;
  std::string card_name;

  std::vector<std::shared_ptr<tensor_info_t>> input;
  std::vector<std::shared_ptr<tensor_info_t>> output;

  std::vector<std::string> input_file;
  std::vector<std::string> output_file;
  std::vector<std::string> param_file;

  std::vector<uint64_t> input_size;
  std::vector<uint64_t> output_size;
  std::vector<uint64_t> param_size;
  uint64_t imm_size;

} Json_common_info_t;

typedef struct chip_common_info {
  uint32_t input_num;
  uint32_t output_num;
  uint32_t param_num;
  uint32_t tile_num;
  uint32_t tile_x;
  uint32_t tile_y;
  std::vector<std::shared_ptr<tensor_info_t>> input;
  std::vector<std::shared_ptr<tensor_info_t>> output;

  // char card_name[100];
  std::string card_name;
  std::vector<std::string> input_file;
  std::vector<std::string> output_file;
  std::vector<std::string> output_ref_file;
  std::vector<std::string> param_file;

  std::vector<uint64_t> input_size;
  std::vector<uint64_t> output_size;
  std::vector<uint64_t> param_size;

  std::vector<TsmHostPtr> input_host_addr;
  std::vector<TsmDevicePtr> input_dev_addr;
  std::vector<TsmHostPtr> output_host_addr;
  std::vector<TsmDevicePtr> output_dev_addr;
  std::vector<TsmHostPtr> param_host_addr;
  std::vector<TsmDevicePtr> param_dev_addr;

  uint64_t imm_size;
} chip_common_info_t;

typedef struct json_common_info_multi_card {
  uint32_t chip_num;
  uint32_t chip_x;
  uint32_t chip_y;
  std::string case_name;
  uint32_t loop_num;
  std::vector<std::shared_ptr<chip_common_info_t>> chip_infos;
} json_common_info_multi_card_t;

typedef struct CompileOption {
  bool comp_enable = false;
  std::string rtt_tool_path;
  std::string compile_path;
  bool check_enable = false;
  uint32_t chip_x;
  uint32_t chip_y;
  bool enable_kcore_bin;
  bool enable_kcore_so;
} CompileOption;

// Boot Param Table
typedef struct BootParamHead {
  uint32_t MaxLen; // BootParamHead + n * BootParamDyninfo, n = inputnum +
                   // outputnum + paramnum
  uint32_t LdmemLen;
  uint32_t InputNum;
  uint32_t OutputNum;
  uint32_t ParamNum;
  uint32_t reserved;
  uint64_t CacheMemLen;
  uint64_t CacheMemAddr; // device
  uint32_t Datalen;
  uint32_t reserved1;
  uint64_t DataAddr; // device
} BootParamHead;

typedef struct BootParamDyninfo {
  uint64_t addr; // device
  uint64_t size;
  uint32_t dtype;
  uint32_t dim;
  uint32_t shape[6]; // #define MAX_SHAPE_DIM      6 //n, h, w, c, x, x
} BootParamDyninfo;

class HrtBootParam {
public:
  HrtBootParam(uint32_t i_num, uint32_t o_num, uint32_t p_num)
      : i_num(i_num), o_num(o_num), p_num(p_num) {
    uint32_t bufsize = (sizeof(BootParamHead) +
                        (i_num + o_num + 1) * sizeof(BootParamDyninfo));
    buffer = (void *)malloc(bufsize);
    memset(buffer, 0, bufsize);
    BootParamHead *head = static_cast<BootParamHead *>(buffer);
    head->MaxLen = bufsize;
    head->LdmemLen = 0x200000;
    head->InputNum = i_num;
    head->OutputNum = o_num;
    head->ParamNum = p_num;
  }
  ~HrtBootParam() {
    if (buffer != nullptr) {
      free(buffer);
    }
  }
  std::vector<BootParamDyninfo> dyninfo;
  uint32_t get_maxlen();
  void *get_bootpmbuffer();
  BootParamHead *get_headptr();
  BootParamDyninfo *get_inputptr(uint32_t index);
  BootParamDyninfo *get_outputptr(uint32_t index);
  BootParamDyninfo *get_paramptr(uint32_t index);
  void set_dev_cache(uint64_t dev_addr, uint64_t size);
  void set_dev_cache_mem_addr(uint64_t dev_addr, uint64_t size);
  void set_dev_dyndata(uint64_t dev_addr, uint32_t size);
  void set_dev_dyndata_mem_addr(uint64_t dev_addr, uint32_t size);
  void set_dev_input(uint32_t idx, uint64_t dev_addr, uint64_t size);
  void set_dev_input_mem_addr(uint32_t idx, uint64_t dev_addr, uint64_t size);
  void set_dev_input_tensor(uint32_t idx,
                            std::shared_ptr<tensor_info_t> tensor);
  void set_dev_output(uint32_t idx, uint64_t dev_addr, uint64_t size);
  void set_dev_output_mem_addr(uint32_t idx, uint64_t dev_addr, uint64_t size);
  void set_dev_param(uint32_t idx, uint64_t dev_addr, uint64_t size);
  void set_dev_param_mem_addr(uint32_t idx, uint64_t dev_addr, uint64_t size);
  std::shared_ptr<tensor_info_t> get_dev_output_tensor_after_run(uint32_t idx);

private:
  uint32_t i_num;
  uint32_t o_num;
  uint32_t p_num;
  void *buffer;
};
/* 启动参数end */

/* compiler后生成的存储elf和param地址的对象 */
class HostParamElem {
public:
  HostParamElem() : dataPtr(nullptr), size(0) {}
  ~HostParamElem();
  // 模拟器:从文件中加载一个bin
  HostParamElem(const std::string &filepath);

  uint8_t *loadBinaryFile(const std::string filepath, uint64_t &fsize);
  uint8_t *dataPtr; // host
  uint64_t size;    // byte
};

class ChipModelInfo {
public:
  ChipModelInfo();
  ChipModelInfo(uint32_t id);
  ~ChipModelInfo();

  uint32_t getChipId() { return chip_id; }
  // support multi chip
  std::vector<std::shared_ptr<HostParamElem>> elfs; // 编译出的elf文件
  std::vector<std::shared_ptr<HostParamElem>> bins; // 编译出的bin文件
  std::vector<std::shared_ptr<HostParamElem>> params;

private:
  uint32_t chip_id;
};

/*
 * compiler后生成的模型对象,在launch的时候会将elf/bin的指针传入soc的接口，
 * (PCIE搬运时，如果空间不连续会触发多次搬运，因此交由SOC组装连续空间。)
 */
class TsmModel {
public:
  TsmModel(); // org_model
  ~TsmModel();
  TsmModel(const std::string &filepath);

  std::vector<std::shared_ptr<ChipModelInfo>> chip_infos;
  THREAD_PROC_FUNC proc_func;
  std::string case_name;
  std::string case_dir;
  std::shared_ptr<HostParamElem> so_list[MAX_MODEL_NUM]
                                        [TILE_MAX_NUM]; // 编译出的so文件
  std::string module_name;
  struct txmodel *model[MAX_MODEL_NUM];
};

typedef struct TsmDevice {
  char res_path[128];
  uint32_t chip_id;
  uint32_t tile_num = 16;
  void *soc_device;
} TsmDevice_t;

class TsmTensorData {
public:
  TsmTensorData() : host_addr(0), device_addr(0), length(0) {}
  ~TsmTensorData(){};

  TsmHostPtr host_addr;
  TsmDevicePtr device_addr;
  uint64_t length;
  uint32_t data_type;
  Tensor_Type tensor_type;
};

typedef void *tsmStream_t;
typedef void *tsmEvent_t;
typedef struct txcclComm *txcclComm_t;
typedef enum { txcclDataDefault = 0 } txcclDataType_t; // 预留，待讨论

enum device_status {
  FULLGOOD = 0,
  PARTIALGOOD = 1,
};

constexpr uint32_t PARTIALGOOD_NUM = 8;
constexpr uint32_t FULLGOOD_NUM = 16;

struct CardComputeInfo {
  uint32_t card_id;
  enum device_status device_status;
  uint32_t all_tile_num;
  double all_tile_compute;
  uint32_t left_tile_num;
  double left_tile_compute;
};

struct TsmDeviceInfo {
  uint32_t card_num;
  uint32_t card_x;
  uint32_t card_y;
  CardComputeInfo card_compute_info[CHIP_MAX_NUM];
};

int32_t readDataFromFile(uint8_t *buffer, std::string file, uint32_t size);
uint8_t *read_file_data(std::string file, uint64_t &size);

std::shared_ptr<json_common_info_multi_card_t>
get_multi_card_common_info_from_file(std::string file);
std::string get_docker_verison();
TSM_RETCODE set_multi_graph(TsmModel *&kmodel,
                            std::shared_ptr<HrtBootParam> &hostboot,
                            const TsmDevicePtr &dev_dyn_mods_ptr,
                            const TsmDevicePtr &dev_tlv_ptr,
                            TsmDevicePtr ext_ptr);
#endif /* __HOST_RUNTIME_COM_H__ */
