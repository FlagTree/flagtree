#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ascend/include/AutoBlockify/Passes.h"
#include "ascend/include/CommonIRToHIVM/ReduceSumStrengthPass.h"
#ifdef __TLE_DSA__
#include "ascend/include/CommonIRToHIVM/Passes.h"
#endif
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/TritonToAnnotation/Passes.h"
#include "ascend/include/TritonToHFusion/Passes.h"
#include "ascend/include/TritonToHIVM/Passes.h"
#include "ascend/include/TritonToLLVM/Passes.h"
#include "incubated/Conversion/DiscreteMaskAccessConversion/Passes.h"
#include "incubated/Conversion/TritonToLinalgIncubated/Passes.h"
#include "incubated/Conversion/TritonToStructuredIncubated/Passes.h"
#include "incubated/Conversion/TritonToUnstructureIncubated/Passes.h"
#include "mlir-ext/Dialect/CommonIR/Transforms/Passes.h"
#include "triton-shared/Conversion/TensorViewLowering/Passes.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewBuilder.h"

#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"
// todo: this code will be removed in version 530.
#include "ascend/include/TritonAffinityOpt/Passes.h"

#include "ir.h" // TritonOpBuilder
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <pybind11/pybind11.h>

#ifdef __TLE_DSA__
#include "llvm/Config/llvm-config.h"
#endif

namespace py = pybind11;
using namespace ir;
using namespace mlir;

static triton::tv::PaddingValue
parseTensorViewPaddingValue(const std::string &value) {
  if (value == "zero")
    return triton::tv::PaddingValue::ZERO;
  if (value == "nan")
    return triton::tv::PaddingValue::NAN_VALUE;
  if (value == "inf")
    return triton::tv::PaddingValue::POS_INF;
  if (value == "-inf")
    return triton::tv::PaddingValue::NEG_INF;
  throw py::value_error("unknown TensorView padding value: " + value);
}

void init_triton_ascend_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls
      ->def("create_extract_scalar",
            [](TritonOpBuilder &self, Value &src,
               std::vector<Value> &indices) -> Value {
              llvm::SmallVector<Value> arg_indices;
              for (const auto &i : indices) {
                auto iTy = i.getType();
                if (!iTy.isIndex()) {
                  auto v = self.create<arith::IndexCastOp>(
                      self.getBuilder().getIndexType(), i);
                  arg_indices.push_back(v);
                } else {
                  arg_indices.push_back(i);
                }
              }
              auto ret = self.create<tensor::ExtractOp>(src, arg_indices);
              return ret;
            })
      .def("create_extract_slice",
           [](TritonOpBuilder &self, Value &ful, std::vector<Value> &offs_vec,
              std::vector<int> &sizs_vec, std::vector<int> &strd_vec) -> Value {
             llvm::SmallVector<Value> offsets;
             llvm::SmallVector<int64_t> staticOffsets;
             for (const auto &o : offs_vec) {
               auto oTy = o.getType();
               if (!oTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), o);
                 offsets.push_back(v);
               } else {
                 offsets.push_back(o);
               }
               staticOffsets.push_back(ShapedType::kDynamic);
             }
             llvm::SmallVector<Value> sizes;
             llvm::SmallVector<int64_t> staticSizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               // auto v = self.create<arith::ConstantIndexOp>(s);
               // sizes.push_back(v);
               staticSizes.push_back(s);
               retSizes.push_back(s);
             }
             llvm::SmallVector<Value> strides;
             llvm::SmallVector<int64_t> staticStrides;
             for (const auto &s : strd_vec) {
               auto v = self.create<arith::ConstantIndexOp>(s);
               strides.push_back(v);
               staticStrides.push_back(ShapedType::kDynamic);
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());

             return self.create<tensor::ExtractSliceOp>(
                 retTy, ful, offsets, sizes, strides, staticOffsets,
                 staticSizes, staticStrides);
           })
      .def("create_insert_slice",
           [](TritonOpBuilder &self, Value &ful, Value &sub,
              std::vector<Value> &offs_vec, std::vector<int> &sizs_vec,
              std::vector<int> &strd_vec) -> Value {
             llvm::SmallVector<Value> offsets;
             llvm::SmallVector<int64_t> staticOffsets;
             for (const auto &o : offs_vec) {
               auto oTy = o.getType();
               if (!oTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), o);
                 offsets.push_back(v);
               } else {
                 offsets.push_back(o);
               }
               staticOffsets.push_back(ShapedType::kDynamic);
             }
             llvm::SmallVector<Value> sizes;
             llvm::SmallVector<int64_t> staticSizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               // auto v = self.create<arith::ConstantIndexOp>(s);
               // sizes.push_back(v);
               staticSizes.push_back(s);
               retSizes.push_back(s);
             }
             llvm::SmallVector<Value> strides;
             llvm::SmallVector<int64_t> staticStrides;
             for (const auto &s : strd_vec) {
               auto v = self.create<arith::ConstantIndexOp>(s);
               strides.push_back(v);
               staticStrides.push_back(ShapedType::kDynamic);
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());
             auto ret = self.create<tensor::InsertSliceOp>(
                 sub, ful, offsets, sizes, strides, staticOffsets, staticSizes,
                 staticStrides);
             return ret;
           })
      .def("create_custom_op_for_inter_core_sync",
           [](TritonOpBuilder &self, std::string &op_name,
              std::string &mode_or_sender, int id) -> void {
             auto args = self.getBuilder().getArrayAttr(
                 {self.getBuilder().getStringAttr(mode_or_sender),
                  self.getBuilder().getI32IntegerAttr(id)});
             self.create<triton::ascend::CustomOp>(op_name, args, ValueRange());
           })
      .def("create_index_select_simd",
           [](TritonOpBuilder &self, Value &src, Value &index, int32_t dim,
              std::vector<Value> &srcShape, std::vector<Value> &srcOffset,
              std::vector<int32_t> &readShape,
              std::vector<int32_t> &returnShape) -> Value {
             auto &builder = self.getBuilder();
             auto loc = self.getLastLoc();

             // Get element type from source pointer
             Type elemType;
             if (auto ptrTy = dyn_cast<triton::PointerType>(src.getType())) {
               elemType = ptrTy.getPointeeType();
             } else {
               llvm::report_fatal_error(
                   "index_select_simd: src must be pointer type");
             }

             // Create return tensor type
             llvm::SmallVector<int64_t> retShape;
             for (const auto &s : returnShape) {
               retShape.push_back(s);
             }
             auto retTensorType = RankedTensorType::get(retShape, elemType);

             // Convert srcShape and srcOffset values to index type if needed
             llvm::SmallVector<Value> srcShapeIndex;
             for (auto val : srcShape) {
               if (!val.getType().isIndex()) {
                 val = self.create<arith::IndexCastOp>(builder.getIndexType(),
                                                       val);
               }
               srcShapeIndex.push_back(val);
             }

             llvm::SmallVector<Value> srcOffsetIndex;
             for (auto val : srcOffset) {
               if (!val.getType().isIndex()) {
                 val = self.create<arith::IndexCastOp>(builder.getIndexType(),
                                                       val);
               }
               srcOffsetIndex.push_back(val);
             }

             // Create attributes
             auto dimAttr = builder.getI32IntegerAttr(dim);
             auto readShapeAttr = builder.getDenseI32ArrayAttr(readShape);

             // Create the IndexSelectSimdOp
             // Parameter order must match TritonOps.td definition:
             // src, index, dim, src_shape, src_offset, read_shape
             auto indexSelectSimdOp =
                 builder.create<triton::ascend::IndexSelectSimdOp>(
                     loc,
                     retTensorType,  // result type
                     src,            // src pointer
                     index,          // index tensor
                     dimAttr,        // dim attribute
                     srcShapeIndex,  // src_shape (variadic, index type)
                     srcOffsetIndex, // src_offset (variadic, index type)
                     readShapeAttr   // read_shape attribute
                 );

             return indexSelectSimdOp.getResult();
           })
      .def("create_index_put",
           [](TritonOpBuilder &self, Value &ptr, Value &index, Value &value,
              const int32_t dim, const int64_t indexBoundary,
              std::vector<Value> &endOffset, std::vector<Value> &startOffset,
              std::vector<Value> &dstStride) -> void {
             // dim need to be i32 type
             auto dimI32Ty = self.getBuilder().getI32Type();
             auto dim_val = self.create<arith::ConstantIntOp>(
                 dimI32Ty, static_cast<int64_t>(dim));
             // indexBoundary need to be i64 type
             auto BoundI64Ty = self.getBuilder().getI64Type();
             auto bound_val = self.create<arith::ConstantIntOp>(
                 BoundI64Ty, static_cast<int64_t>(indexBoundary));

             self.create<triton::ascend::IndexPutOp>(ptr, index, value, dim_val,
                                                     bound_val, endOffset,
                                                     startOffset, dstStride);
           })
      .def("create_gather_out_to_ub",
           [](TritonOpBuilder &self, Value &src, Value &index,
              const int64_t indexBoundary, const int32_t dim,
              std::vector<Value> &srcStride, std::vector<Value> &endOffset,
              std::vector<Value> &startOffset,
              std::optional<Value> &other) -> Value {
             auto elemTy = cast<PointerType>(src.getType()).getPointeeType();
             auto idxTy = cast<RankedTensorType>(index.getType());
             auto idxShape = idxTy.getShape();
             std::vector<int64_t> retShape(idxShape.begin(), idxShape.end());
             auto resType = RankedTensorType::get(retShape, elemTy);

             // indexBoundary need to be i64 type
             auto BoundI64Ty = self.getBuilder().getI64Type();
             auto bound_val = self.create<arith::ConstantIntOp>(
                 BoundI64Ty, static_cast<int64_t>(indexBoundary));
             // dim need to be i32 type
             auto dimI32Ty = self.getBuilder().getI32Type();
             auto dim_val = self.create<arith::ConstantIntOp>(
                 dimI32Ty, static_cast<int64_t>(dim));
             return self.create<triton::ascend::GatherOutToUbOp>(
                 resType, src, index, bound_val, dim_val, srcStride, endOffset,
                 startOffset, other.value_or(Value()));
           })
      .def("create_scatter_ub_to_out",
           [](TritonOpBuilder &self, Value &ptr, Value &value, Value &index,
              const int64_t indexBoundary, const int32_t dim,
              std::vector<Value> &dstStride, std::vector<Value> &endOffset,
              std::vector<Value> &startOffset) -> void {
             auto idxTy = cast<RankedTensorType>(index.getType());

             // indexBoundary need to be i64 type
             auto BoundI64Ty = self.getBuilder().getI64Type();
             auto bound_val = self.create<arith::ConstantIntOp>(
                 BoundI64Ty, static_cast<int64_t>(indexBoundary));
             // dim need to be i32 type
             auto dimI32Ty = self.getBuilder().getI32Type();
             auto dim_val = self.create<arith::ConstantIntOp>(
                 dimI32Ty, static_cast<int64_t>(dim));

             self.create<triton::ascend::ScatterUbToOutOp>(
                 ptr, value, index, bound_val, dim_val, dstStride, endOffset,
                 startOffset);
           })
      // conv1d operation
      .def(
          "create_conv1d",
          [](TritonOpBuilder &self, Value input, Value weight, py::object bias,
             int64_t stride, int64_t padding_size, int64_t dilation,
             int64_t groups, Type output_type) -> Value {
            Value biasValue;
            if (!bias.is_none()) {
              biasValue = bias.cast<Value>();
            } else {
              biasValue = Value();
            }
            auto &builder = self.getBuilder();
            auto strideAttr = builder.getI64IntegerAttr(stride);
            auto paddingSizeAttr = builder.getI64IntegerAttr(padding_size);
            auto dilationAttr = builder.getI64IntegerAttr(dilation);
            auto groupsAttr = builder.getI64IntegerAttr(groups);
            auto op = self.create<triton::ascend::Conv1dOp>(
                output_type, input, weight, biasValue, strideAttr,
                paddingSizeAttr, dilationAttr, groupsAttr);
            return op.getResult();
          },
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding_size"), py::arg("dilation"),
          py::arg("groups"), py::arg("output_type"))
      // Add sort
      .def("create_sort",
           [](TritonOpBuilder &self, Value src, int64_t dim,
              bool descending) -> Value {
             auto &builder = self.getBuilder();
             auto loc = self.getLastLoc();

             auto dimAttr = builder.getI64IntegerAttr(dim);
             auto descendingAttr = builder.getBoolAttr(descending);

             auto op = builder.create<triton::ascend::SortOp>(loc, src, dimAttr,
                                                              descendingAttr);

             return op->getResult(0);
           })
      // Add flip
      .def("create_flip",
           [](TritonOpBuilder &self, Value src, int64_t dim) -> Value {
             auto &builder = self.getBuilder();
             auto loc = self.getLastLoc();

             auto dimAttr = builder.getI64IntegerAttr(dim);

             auto op =
                 builder.create<triton::ascend::FlipOp>(loc, src, dimAttr);

             return op->getResult(0);
           })
      .def("create_annotation",
           [](TritonOpBuilder &self, Value &ptr, const std::string &attrKey,
              Attribute &attrVal) {
             auto annotationOp = self.create<triton::ascend::AnnotationOp>(ptr);
             annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                                   attrVal);
           })
      // Convert or bridge a call argument to !tv.ptr.
      .def("convert_to_tensor_view_ptr",
           [](TritonOpBuilder &self, Value value) -> Value {
             return triton::tv::convertToTensorViewPtr(value);
           })
      //===----------------------------------------------------------------===//
      // Direct-construction API for `tl.tensor_view`.
      //===----------------------------------------------------------------===//
      .def("create_tensor_view",
           [](TritonOpBuilder &self, Value base, std::vector<Value> shape,
              std::vector<Value> strides) -> Value {
             FailureOr<Value> view = triton::tv::createTensorViewBase(
                 self.getBuilder(), self.getLastLoc(), base, shape, strides);
             if (failed(view))
               throw py::value_error(
                   "TensorView base pointers must not contain pointer offsets");
             return *view;
           })
      .def("create_partition_view",
           [](TritonOpBuilder &self, Value view,
              std::vector<int64_t> tile, std::string paddingValue) -> Value {
             return triton::tv::createPartitionView(
                 self.getBuilder(), self.getLastLoc(), view, tile,
                 parseTensorViewPaddingValue(paddingValue));
           })
      .def("create_strided_view",
           [](TritonOpBuilder &self, Value view, std::vector<int64_t> tile,
              std::vector<int64_t> traversalStrides,
              std::string paddingValue) -> Value {
             return triton::tv::createStridedView(self.getBuilder(),
                                                  self.getLastLoc(), view, tile,
                                                  traversalStrides,
                                                  parseTensorViewPaddingValue(
                                                      paddingValue));
           })
      .def("create_gather_scatter_view",
           [](TritonOpBuilder &self, Value view, std::vector<int64_t> tile,
              std::vector<int64_t> sparseDims,
              std::string paddingValue) -> Value {
             return triton::tv::createGatherScatterView(
                 self.getBuilder(), self.getLastLoc(), view, tile, sparseDims,
                 parseTensorViewPaddingValue(paddingValue));
           })
      .def("tensor_view_load",
           [](TritonOpBuilder &self, Value view, std::vector<Value> index,
              Type resultTy) -> Value {
             return triton::tv::tensorViewLoad(
                 self.getBuilder(), self.getLastLoc(), view, index, resultTy);
           })
      .def("tensor_view_store",
           [](TritonOpBuilder &self, Value view, Value value,
              std::vector<Value> index) -> void {
             triton::tv::tensorViewStore(self.getBuilder(), self.getLastLoc(),
                                         view, index, value);
           })
      .def("tensor_view_ptr_load",
           [](TritonOpBuilder &self, Value pointers,
              const std::vector<std::vector<int64_t>> &coordinates,
              Type resultTy) -> Value {
             llvm::SmallVector<llvm::ArrayRef<int64_t>> coordinateRefs;
             for (const auto &coordinate : coordinates)
               coordinateRefs.push_back(coordinate);
             return triton::tv::tensorViewPtrLoad(self.getBuilder(),
                                                  self.getLastLoc(), pointers,
                                                  coordinateRefs, resultTy);
           })
      .def("tensor_view_ptr_store",
           [](TritonOpBuilder &self, Value pointers, Value value,
              const std::vector<std::vector<int64_t>> &coordinates) -> void {
             llvm::SmallVector<llvm::ArrayRef<int64_t>> coordinateRefs;
             for (const auto &coordinate : coordinates)
               coordinateRefs.push_back(coordinate);
             triton::tv::tensorViewPtrStore(self.getBuilder(),
                                            self.getLastLoc(), pointers, value,
                                            coordinateRefs);
           })
      // Build the TensorView type used across IR boundaries.
      .def("get_tensor_view_ty",
           [](TritonOpBuilder &self, Type elementType, int64_t rank,
              std::string viewKind, std::vector<int64_t> tile,
              std::vector<int64_t> traversalStrides,
              std::vector<int64_t> sparseDims,
              std::string paddingValue) -> Type {
             llvm::SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
             llvm::SmallVector<int64_t> dynStrides(rank, ShapedType::kDynamic);
             Attribute encoding;
             triton::tv::PaddingValue padding =
                 parseTensorViewPaddingValue(paddingValue);
             if (viewKind == "partition") {
               encoding = triton::tv::PartitionViewAttr::get(
                   self.getBuilder().getContext(), tile, padding);
             } else if (viewKind == "strided") {
               encoding = triton::tv::StridedViewAttr::get(
                   self.getBuilder().getContext(), tile, traversalStrides,
                   padding);
             } else if (viewKind == "gather_scatter") {
               encoding = triton::tv::GatherScatterViewAttr::get(
                   self.getBuilder().getContext(), tile, sparseDims, padding);
             } else if (!viewKind.empty()) {
               throw py::value_error("unknown TensorView encoding: " +
                                     viewKind);
             }
             return triton::tv::TensorViewType::get(dynShape, elementType,
                                                    dynStrides, encoding);
           });
}

void init_triton_ascend_passes_ttir(py::module &&m) {
  m.def("add_lane_vectorize", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::tile::createTritonLaneVectorize());
  });

  m.def("add_auto_blockify", [](mlir::PassManager &pm, int autoBlockifySize) {
    AutoBlockifyOptions opts;
    opts.autoBlockifySize = autoBlockifySize;
    pm.addPass(mlir::triton::createAutoBlockifyPass(opts));
  });

  m.def("add_reduce_sum_strength", [](mlir::PassManager &pm, bool enable,
                                      int32_t splitFactor) {
    pm.addPass(mlir::triton::createReduceSumStrengthPass(enable, splitFactor));
  });

  m.def("add_triton_to_structure",
        [](mlir::PassManager &pm, bool enableMaskFallbackConversion,
           bool optimizeDynamicOffset) {
          pm.addPass(mlir::triton::createTritonToStructuredIncubatedPass(
              enableMaskFallbackConversion, optimizeDynamicOffset));
        });

  m.def("add_triton_to_annotation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToAnnotationPass());
  });

  m.def("add_triton_to_linalg",
        [](mlir::PassManager &pm, bool globalKernel, bool namedOps,
           bool enableNd2nzOnVector, bool enableSelectAnalysis,
           bool compileOn91095) {
          pm.addPass(mlir::triton::Incubated::createTritonToLinalgIncubatedPass(
              globalKernel, namedOps, enableNd2nzOnVector, enableSelectAnalysis,
              compileOn91095));
        });

  m.def("add_triton_to_unstructure",
        [](mlir::PassManager &pm, bool compileOn91095, bool forceSimtTemplate) {
          TritonToUnstructureIncubatedOptions opts;
          opts.compileOn91095 = compileOn91095;
          opts.forceSimtTemplate = forceSimtTemplate;
          pm.addPass(
              mlir::triton::createTritonToUnstructureIncubatedPass(opts));
        });

  m.def("add_triton_to_hfusion", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToHFusionPass());
  });

  m.def("add_discrete_mask_access_conversion", [](mlir::PassManager &pm,
                                                  bool compileOn91095,
                                                  bool forceSimtTemplate,
                                                  bool enableSyncBlockLock) {
    DiscreteMaskAccessConversionOptions opts;
    opts.compileOn91095 = compileOn91095;
    opts.forceSimtTemplate = forceSimtTemplate;
    opts.enableSyncBlockLock = enableSyncBlockLock;
    pm.addPass(mlir::triton::createDiscreteMaskAccessConversionPass(opts));
  });

  m.def("add_triton_to_hivm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToHIVMPass());
  });

  m.def("add_tensor_view_lowering", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTensorViewLoweringPass());
  });

#ifdef __TLE_DSA__
  m.def("add_commonir_tv_ptr_legalize", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createCommonIRTvPtrLegalizePass());
  });

  m.def("add_commonir_to_hivm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createCommonIRToHIVMPass());
  });
#endif

  m.def("add_triton_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToLLVMPass());
  });

  m.def("add_bubble_up_operation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createBubbleUpOperationPass());
  });

  m.def("add_dynamic_cv_pipeline",
        [](mlir::PassManager &pm, bool compileOn91095) {
          AddDynamicCVPipelineOptions opts;
          opts.compileOn91095 = compileOn91095;
          pm.addPass(mlir::triton::createAddDynamicCVPipelinePass(opts));
        });

  // todo: this code will be removed in version 530.
  m.def("add_dag_sync", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createDAGSyncPass());
  });

  m.def("add_dag_scope", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createDAGScopePass());
  });

  m.def("add_dag_ssbuffer", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createDAGSSBufferPass());
  });

  m.def("set_buffer_count", [](int type, int count) {
    auto depType = static_cast<mlir::triton::BufferCountManager::DepType>(type);
    mlir::triton::BufferCountManager::getInstance().setBufferCount(depType,
                                                                   count);
  });
}

// Forward declaration for ascend_ir bindings (defined in ascend_ir.cc)
void init_ascend_ir(py::module &&m);

void init_triton_ascend(py::module &&m) {
  auto passes = m.def_submodule("passes");
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::ascend::TritonAscendDialect,
                    mlir::triton::tv::TensorViewDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_ascend_passes_ttir(passes.def_submodule("ttir"));
  init_triton_ascend_ir(m.def_submodule("ascend_ir"));

  // Initialize ascend IR bindings (ascendnpu_ir_builder, scope/hivm dialects)
  init_ascend_ir(m.def_submodule("ir"));
}
