# should store at third_party/nvidia/backend/
from triton.compiler.hint_manager import BaseHintHandler


class NvidiaHintHandler(BaseHintHandler):

    @staticmethod
    def inject_kwargs_with_hints(fn, flagtree_hints, line_num, kws):
        if fn.__name__ == "load" and flagtree_hints is not None:
            print(f"[FLAGTREE] tl.load at line {line_num} has annotation {flagtree_hints}")
            if 'flagtree_hints' not in kws:
                kws['flagtree_hints'] = ""
            if flagtree_hints not in kws['flagtree_hints']:
                kws['flagtree_hints'] = flagtree_hints
