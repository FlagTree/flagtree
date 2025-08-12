def language_extra_cuda_get_all():
    __all__ = [
                "libdevice",
                #"globaltimer",
                "num_threads", "num_warps",
                #"smid",
                "convert_custom_float8_sm70", "convert_custom_float8_sm80"
            ]
    return __all__