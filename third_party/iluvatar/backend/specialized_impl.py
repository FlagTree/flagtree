from triton.runtime.flagtree_backend_manager import register_backend


class IluvatarSpecializedImpl():

    @staticmethod
    def kernel_suffix(signature, specialization):
        # suffix format:
        # <argid><'c' if equal to 1><'d' if divisible by 16><'e' if divisible by 8>
        suffix = ''
        for i, _ in enumerate(signature):
            suffix += str(i)
            if i in specialization.equal_to_1:
                suffix += 'c'
            if i in specialization.divisible_by_16:
                suffix += 'd'
            if i in specialization.divisible_by_8:
                suffix += 'e'
        return suffix

    @staticmethod
    def ast_to_ttir(fn, specialization, context, options, codegen_fns):
        from triton.compiler.code_generator import _get_fn_file_line, CodeGenerator
        from triton.language import str_to_ty
        from triton import language

        attrs = specialization.attrs
        # create kernel prototype
        cst_key = lambda i: fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in specialization.constants.items()}
        # visit kernel AST
        gscope = fn.__globals__.copy()
        function_name = fn.repr(specialization)
        tys = list(specialization.signature.values())
        new_constants = {k: True if k in tys and tys[k] == "i1" else 1 for k in attrs.equal_to_1}
        new_attrs = {k[0]: [("tt.corex_stride", k[1])] for k in attrs.corexLoad.items()}
        for k in attrs.divisible_by_16:
            attr = new_attrs[k] if k in new_attrs else []
            attr.append(("tt.divisibility", 16))
            new_attrs[k] = attr

        all_constants = constants.copy()
        all_constants.update(new_constants)
        arg_types = [str_to_ty(v) for k, v in specialization.signature.items() if k not in specialization.constants]
        file_name, begin_line = _get_fn_file_line(fn)

        prototype = language.function_type([], arg_types)
        generator = CodeGenerator(context, prototype, gscope=gscope, constants=all_constants,
                                  function_name=function_name, jit_fn=fn, attributes=new_attrs, is_kernel=True,
                                  file_name=file_name, begin_line=begin_line, options=options, codegen_fns=codegen_fns)
        generator.visit(fn.parse())

        ret = generator.module
        # module takes ownership of the context
        ret.context = context
        return ret


register_backend("iluvatar", IluvatarSpecializedImpl())
