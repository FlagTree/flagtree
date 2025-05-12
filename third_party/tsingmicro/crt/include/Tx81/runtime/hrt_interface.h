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

#ifndef __HOST_RUNTIME_INTERFACE_H__
#define __HOST_RUNTIME_INTERFACE_H__

#include <string>
#include <vector>

#include "hrt_common.h"

/*
 *  以下接口依赖Runtime实例生命周期中，即调用TsmInitRuntime后，调用TsmDeInitRuntime前
 */
TSM_RETCODE TsmInitRuntime(void);
TSM_RETCODE TsmDeInitRuntime(void);
TSM_RETCODE TsmDeInitRuntimeLegacy(void);
TSM_RETCODE TsmSetDevice(uint32_t first_phy_id, uint32_t card_x,
                         uint32_t card_y, std::vector<TsmDevice *> &devs);
TSM_RETCODE TsmSetDeviceOld(
    uint32_t chip_id,
    TsmDevice *dev); /* 该接口为提供给MLIR的过度版本，其他组件不要调用 */
TSM_RETCODE TsmDeviceMalloc(TsmDevice *dev, TsmDevicePtr &ptr, uint64_t size);
TSM_RETCODE TsmDeviceMemset(TsmDevicePtr &ptr, uint32_t ch, uint64_t size);
TSM_RETCODE TsmDeviceFree(TsmDevicePtr ptr);
TSM_RETCODE TsmDeviceSynchronize(TsmDevice *dev);
TSM_RETCODE TsmInitDevice(TsmDevice *dev);
TSM_RETCODE TsmCompile(std::vector<TsmDevice *> devs, TsmModel &kmodel,
                       std::string option, CompileOption compl_op);
TSM_RETCODE TsmCompileMultiGraph(std::vector<TsmDevice *> devs,
                                 TsmModel &kmodel, std::string option,
                                 CompileOption compl_op);
TSM_RETCODE TsmLaunch(TsmDevice *dev, TsmModel &kmodel);
TSM_RETCODE TsmLoadKernel(TsmDevice *dev, std::vector<TsmModel *> &kmodel_vec,
                          char *module_symbol);
TSM_RETCODE TsmUnloadKernel(TsmDevice *dev,
                            std::vector<TsmModel *> &kmodel_vec);
TSM_RETCODE TsmRun(TsmDevice *dev, TsmDevicePtr bootpm_dev);
TSM_RETCODE TsmAsyncRun(tsmStream_t stream, TsmDevice *dev,
                        TsmDevicePtr bootpm_dev);
TSM_RETCODE TsmSetTerminate(TsmDevice *dev, tsmStream_t stream = nullptr);
TSM_RETCODE TsmGetDeviceInfo(TsmDeviceInfo *info);
TSM_RETCODE TsmTerminate(TsmDevice *dev, TsmDevicePtr bootpm_dev);
TSM_RETCODE TsmMemcpyH2D(TsmDevicePtr dst, const void *src,
                         uint64_t byte_count);
TSM_RETCODE TsmMemcpyD2H(const void *dst, TsmDevicePtr src,
                         uint64_t byte_count);
TSM_RETCODE TsmMemcpyOffsetH2D(TsmDevicePtr dst, const void *src,
                               uint64_t offset, uint64_t byte_count);
TSM_RETCODE TsmMemcpyOffsetD2H(const void *dst, TsmDevicePtr src,
                               uint64_t offset, uint64_t byte_count);
TSM_RETCODE TsmMemcpyD2D(const void *dst, TsmDevice *dst_dev, const void *src,
                         TsmDevice *src_dev, uint64_t byte_count);
TSM_RETCODE TsmSend(const void *sendbuff, size_t count,
                    txcclDataType_t datatype, TsmDevice *dev, int peer,
                    txcclComm_t comm, tsmStream_t stream);
TSM_RETCODE TsmRecv(void *recvbuff, size_t count, txcclDataType_t datatype,
                    TsmDevice *dev, int peer, txcclComm_t comm,
                    tsmStream_t stream);
TSM_RETCODE TsmResetDevice(TsmDevice *dev);
TSM_RETCODE TsmReleaseDevice(TsmDevice *dev);
TSM_RETCODE TsmMemGetInfo(TsmDevicePtr ptr, uint32_t &card_id, uint64_t &addr,
                          uint64_t &size);
TSM_RETCODE TsmEventCreate(tsmEvent_t *pEvent);
TSM_RETCODE TsmEventDestroy(tsmEvent_t event);
TSM_RETCODE TsmEventRecord(tsmEvent_t event, tsmStream_t stream);
TSM_RETCODE TsmEventWait(tsmEvent_t event, tsmStream_t stream);
TSM_RETCODE TsmStreamCreate(tsmStream_t *pStream, TsmDevice *dev);
TSM_RETCODE TsmStreamSynchronize(tsmStream_t stream);
TSM_RETCODE TsmStreamDestroy(tsmStream_t stream);
TSM_RETCODE TsmDeviceSerialize(const TsmDevice *const &dev, void *&buffer,
                               size_t &size);
TSM_RETCODE TsmDeviceDeSerialize(TsmDevice *&dev, const void *const &buffer);
TSM_RETCODE TsmSetMonitorInfo(TsmDevice *dev);
TSM_RETCODE TsmProcessProfData(TsmDevice *dev, TsmProfAction prof_action,
                               uint16_t prof_type);
TSM_RETCODE TsmHostH2D(TsmDevice *dev, uint64_t input_host_addr,
                       uint64_t input_size, int32_t index);
TSM_RETCODE TsmHostFlush(TsmDevice *dev, uint64_t boot_param_ptr,
                         uint8_t *host_buffer, size_t size);
TSM_RETCODE TsmSetRankSize(uint32_t x_size, uint32_t y_size);
TSM_RETCODE TsmSetRankId(uint32_t x, uint32_t y);
TSM_RETCODE TsmGetPhyRankId(uint32_t *x, uint32_t *y);

/*
 *  以下接口为无状态，不依赖Runtime实例，可以独立使用
 */
TSM_RETCODE TsmGetDeviceNum(uint32_t &dev_num);

/*
 * 为保持Host日志格式统一，Runtime提供了统一日志接口，各组件按以下方式使用：
 * #define rt_log(level, format, ...) tsm_log(__FILE__, __func__, __LINE__,
 * TSM_RUNTIME, level, format, ##__VA_ARGS__)
 *
 * void func() {
 *   rt_log(LOG_DEBUG, "....\n");
 *   rt_log(LOG_INFO, "....\n");
 *   rt_log(LOG_WARNING, "....\n");
 *   rt_log(LOG_ERROR, "....\n");
 * }
 * 默认日志级别为INFO，通过设置 HOST_LOG_LEVEL
 * 更改日志级别，一般就设置成INFO和DEBUG。 注意：
 *    其中rt_log为各组件定制名称，切勿重复，TSM_RUNTIME表示模块ID，各模块到hrt_common.h找到自己的宏，没有的可以联系runtime来增加。
 */
void tsm_log(const char *file_name, const char *func_name, uint32_t line_number,
             TsmModuleType module_type, HostLogLevel level, const char *format,
             ...);
#endif
