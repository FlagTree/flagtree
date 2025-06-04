#ifndef TRITON_DEVICE_PRINT_H
#define TRITON_DEVICE_PRINT_H

#include "experiment/runtime/runtime/rt.h"
#include "stdio.h"

#define LogBufferPaddingBytes 64
#define BlockMaxSize 16 * 1024
#define VerifyBorder(nextField, maxBuf)                                        \
  if (nextField > maxBuf) {                                                    \
    printf("\nWARNING: out of bound! try best to print\n");                    \
    return;                                                                    \
  }
#define __gm__

namespace TTAscDebug {

enum NodeTy { END, NORMAL, FLOAT, INT, CHAR, STRING, POINTER };

struct PrintPayloadData {
  __gm__ char *LogWholeRegion;
  unsigned BlockNum;
  size_t LogBufferSize;
  PrintPayloadData()
      : LogWholeRegion((__gm__ char *)nullptr), LogBufferSize(0), BlockNum(0) {}
};

struct DebugTunnelData {
  PrintPayloadData PrintData;
  DebugTunnelData() {}
};

void PrintFormatString(int8_t *&buf, int8_t *maxbuf) {
  VerifyBorder((buf + sizeof(short)), maxbuf);
  short len = *(short *)buf;
  buf += sizeof(len);
  VerifyBorder((buf + len), maxbuf);
  printf((const char *)buf);
  buf += len;
}

template <typename T>
void PrintFormatString(int8_t *&buf, int8_t *maxbuf, T param) {
  VerifyBorder((buf + sizeof(short)), maxbuf);
  short len = *(short *)buf;
  buf += sizeof(len);
  VerifyBorder((buf + len), maxbuf);
  printf((const char *)buf, param);
  buf += len;
}

void AnalyzeSerializedData(int8_t *buf, int logSize, int maxSize) {
  int8_t *bufEndAddr = buf + logSize;
  int8_t *maxbuf = buf + maxSize;
  while (buf < bufEndAddr) {
    VerifyBorder((buf + sizeof(int8_t)), maxbuf);
    int8_t type = *(int8_t *)buf;
    while (type != NodeTy::END) {
      buf += sizeof(type);
      switch (type) {
      default:
        break;
      case NodeTy::NORMAL: {
        PrintFormatString(buf, maxbuf);
        break;
      }
      case NodeTy::FLOAT: {
        VerifyBorder((buf + sizeof(float)), maxbuf);
        float param = *(float *)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case NodeTy::INT: {
        VerifyBorder((buf + sizeof(long long int)), maxbuf);
        long long int param = *(long long int *)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case NodeTy::STRING: {
        VerifyBorder((buf + sizeof(short)), maxbuf);
        short strlen = *(short *)buf;
        buf += sizeof(strlen);
        VerifyBorder((buf + strlen), maxbuf);
        char *param = reinterpret_cast<char *>(buf);
        buf += strlen;
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case NodeTy::CHAR: {
        VerifyBorder((buf + sizeof(char)), maxbuf);
        char param = *(char *)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case NodeTy::POINTER: {
        VerifyBorder((buf + 8), maxbuf);
        void *param = *(void **)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      }
      VerifyBorder((buf + sizeof(int8_t)), maxbuf);
      type = *(int8_t *)buf;
    }
    buf += 1;
  }
}

void OnHostInitialize(PrintPayloadData *PrintData, unsigned BlockNum) {
  PrintData->LogBufferSize = BlockMaxSize;
  PrintData->BlockNum = BlockNum;
  int WholeSize =
      (PrintData->LogBufferSize + LogBufferPaddingBytes) * PrintData->BlockNum;

  void *Hbm_PrintPayloadData_start_addr = NULL;
  // Not sure how to use the module_id param of rtMalloc
  uint16_t ModuleId = 0;
  rtError_t error =
      rtMalloc(reinterpret_cast<void **>(&Hbm_PrintPayloadData_start_addr),
               WholeSize, RT_MEMORY_HBM, ModuleId);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:The memory for the printing function on the device side "
           "fails to be allocated.");
    printf("As a result, the printing function fails!\n");
    return;
  }
  PrintData->LogWholeRegion = (__gm__ char *)Hbm_PrintPayloadData_start_addr;
}

void OnHostFinish(PrintPayloadData *PrintData, rtStream_t Stream) {
  if (!PrintData->LogWholeRegion) {
    return;
  }
  std::size_t WholeSize =
      (PrintData->LogBufferSize + LogBufferPaddingBytes) * PrintData->BlockNum;
  char *hostMemOut2;
  // Not sure how to use the module_id param of rtMalloc
  uint16_t ModuleId = 0;
  rtError_t error = rtMallocHost(reinterpret_cast<void **>(&hostMemOut2),
                                 WholeSize, ModuleId);
  if (error != RT_ERROR_NONE) {
    printf("ERROR:The memory for the printing function on the device side "
           "fails to be allocated.");
    printf("As a result, the printing function fails!\n");
    return;
  }
  error = rtMemcpyAsync(hostMemOut2, WholeSize, PrintData->LogWholeRegion,
                        WholeSize, RT_MEMCPY_DEVICE_TO_HOST, Stream);
  if (error != RT_ERROR_NONE) {
    printf("ERROR: The memory copy of the device print on fails,");
    printf("and the printing function is invalid!\n");
    return;
  }
  error = rtStreamSynchronize(Stream);
  if (error != RT_ERROR_NONE) {
    printf("ERROR: Synchronous waiting for the device print failed.\n");
    printf("The printing function is invalid!\n");
    return;
  }
  char *outRaw2 = static_cast<char *>(hostMemOut2);
  const char *Line = "-------------------------------------------------------";
  // Precheck if any print data is ready
  for (int B = 0; B < PrintData->BlockNum; B++) {
    char *Log =
        (outRaw2 + (PrintData->LogBufferSize + LogBufferPaddingBytes) * B);
    size_t LogSize = *reinterpret_cast<size_t *>(Log);
    if (LogSize > 0 && LogSize <= PrintData->LogBufferSize) {
      printf("LogBufferSize of each core is : %zu Bytes\n",
             PrintData->LogBufferSize);
      printf("%s\n", Line);
      printf("----------------------HiIPU "
             "Print----------------------\n");
      printf("%s\n", Line);
      break;
    }
  }

  for (int B = 0; B < PrintData->BlockNum; B++) {
    char *Log =
        (outRaw2 + (PrintData->LogBufferSize + LogBufferPaddingBytes) * B);
    size_t LogSize = *reinterpret_cast<size_t *>(Log);
    if (LogSize < 0 || LogSize > PrintData->LogBufferSize) {
      printf(" LOG SIZE ERROR !!! \n");
      printf(" log size needed = %zu ", LogSize);
      printf(" , buf size = %zu\n", PrintData->LogBufferSize);
      LogSize = PrintData->LogBufferSize;
      continue;
    }
    if (LogSize == 0) {
      continue;
    }
    printf("==> Block %d, LogSize = %zu Bytes\n", B, LogSize);
    int8_t *Buf = reinterpret_cast<int8_t *>(Log + LogBufferPaddingBytes);
    AnalyzeSerializedData(Buf, LogSize, PrintData->LogBufferSize);
    printf("\n");
    printf("%s\n", Line);
  }
  error = rtFree(PrintData->LogWholeRegion);
  if (error != RT_ERROR_NONE) {
    printf("ERROR: The memory free of the device print fails\n");
    return;
  }
  error = rtFreeHost(hostMemOut2);
  if (error != RT_ERROR_NONE) {
    printf("ERROR: The memory free of the device print fails\n");
    return;
  }
}

DebugTunnelData *Open(unsigned BlockNum) {
  DebugTunnelData debugTunnelDataForHost;
  OnHostInitialize(&(debugTunnelDataForHost.PrintData), BlockNum);
  void *Hbm_PrintPayloadData_start_addr = NULL;
  // Not sure how to use the module_id param of rtMalloc
  uint16_t ModuleId = 0;
  rtError_t error =
      rtMalloc(reinterpret_cast<void **>(&Hbm_PrintPayloadData_start_addr),
               sizeof(debugTunnelDataForHost), RT_MEMORY_HBM, ModuleId);
  if (error != RT_ERROR_NONE) {
    printf("ERROR: The memory for the printing function on the device side "
           "fails to be allocated.");
    printf("As a result, the printing function fails!\n");
    return nullptr;
  }
  if (Hbm_PrintPayloadData_start_addr == nullptr) {
    printf("WARNING: failed to allocate DebugTunnelData memory\n");
    return nullptr;
  }
  error = rtMemcpy(Hbm_PrintPayloadData_start_addr,
                   sizeof(debugTunnelDataForHost), &debugTunnelDataForHost,
                   sizeof(debugTunnelDataForHost), RT_MEMCPY_HOST_TO_DEVICE);
  if (error != RT_ERROR_NONE) {
    printf("ERROR: The memory copy of the device print on fails, ");
    printf("and the printing function is invalid!\n");
    return nullptr;
  }
  return reinterpret_cast<DebugTunnelData *>(Hbm_PrintPayloadData_start_addr);
}

void Close(DebugTunnelData *DTData, rtStream_t Stream) {
  if (!DTData) {
    return;
  }
  DebugTunnelData debugTunnelDataForHost;
  rtError_t error = rtStreamSynchronize(Stream);
  if (error != RT_ERROR_NONE) {
    printf("ERROR: Synchronous waiting for the device print failed.\n");
    printf("The printing function is invalid!\n");
  }
  error =
      rtMemcpy(&debugTunnelDataForHost, sizeof(debugTunnelDataForHost), DTData,
               sizeof(debugTunnelDataForHost), RT_MEMCPY_DEVICE_TO_HOST);
  if (error != RT_ERROR_NONE) {
    printf("ERROR: The memory copy of the device print on fails, ");
    printf("and the printing function is invalid!\n");
    return;
  }
  OnHostFinish(&(debugTunnelDataForHost.PrintData), Stream);

  error = rtFree(DTData);
  if (error != RT_ERROR_NONE) {
    printf("ERROR: The memory free of the device print fails, ");
    printf("and the device print is invalid!\n");
    return;
  }
}

} // namespace TTAscDebug

#endif
