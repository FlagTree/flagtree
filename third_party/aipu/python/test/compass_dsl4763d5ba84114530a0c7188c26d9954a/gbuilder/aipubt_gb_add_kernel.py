import numpy as np
from AIPUBuilder.core import BuilderOpPlugin, register_optype, BuilderParams, Tensor
from AIPUBuilder.plugin_loader import register_plugin, PluginType


op_type = register_optype("DSL_add_kernel")

def _get_addr(base, offset_in_byte):
    if offset_in_byte == 0:
        return base

    ret = Tensor(base)
    ret.memory_offset().set_base_offset(base.memory_offset())
    ret.memory_offset().relative_offset += offset_in_byte
    return ret

@register_plugin(PluginType.Builder, 0)
class DSL_add_kernelPlugin(BuilderOpPlugin):
    def get_graph_pattern(self):
        return ([("useless", op_type)], [])

    def get_score(self):
        return 10

    def set_target(self, target):
        return True

    def check_params(self, nodes):
        return True

    def setup(self, sgnode, nodes):
        sgnode.attrs["keeping_layout"] = False
        return True

    def generate_code_name(self, sgnode, nodes):
        return "op_lib/add_kernel.o"

    def generate_descriptor(self, sgnode, nodes):
        desc_base = sgnode.attrs["descriptorbase"]
        ro = BuilderParams()
        return ro

    def generate_params(self, sgnode, nodes):
        desc_base = sgnode.attrs["descriptorbase"]
        ro = BuilderParams()
        ro.append(sgnode.inputs[0])
        ro.append(sgnode.inputs[1])
        ro.append(sgnode.outputs[0])
        value = np.int32(nodes[0].params["var_14"])
        ro.append(int(value.view("int32")))
        value = np.int32(nodes[0].params["var_33"])
        ro.append(int(value.view("int32")))
        value = np.int32(nodes[0].params["var_35"])
        ro.append(int(value.view("int32")))
        value = np.int32(nodes[0].params["var_37"])
        ro.append(int(value.view("int32")))
        value = np.int32(nodes[0].params["var_3"])
        ro.append(int(value.view("int32")))
        value = np.int32(nodes[0].params["var_39"])
        ro.append(int(value.view("int32")))
        value = np.int32(nodes[0].params["var_41"])
        ro.append(int(value.view("int32")))
        return ro
