from functools import wraps
import logging
import sys

logger = logging.getLogger(__name__)


def hex_to_rgba(hex_color: str) -> tuple[int, ...]:
    """
    Converts a hex color string to an ARGB tuple.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        return (*(int(hex_color[i : i + 2], 16) for i in (0, 2, 4)), 255)
    elif len(hex_color) == 8:
        return tuple(int(hex_color[i : i + 2], 16) for i in (2, 4, 6, 0))
    else:
        raise ValueError(f"Hex color must be 6 or 8 digits. Got {hex_color}.")


class TraceCalls(object):
    """Use as a decorator on functions that should be traced. Several
    functions can be decorated - they will all be indented according
    to their call depth.
    """

    def __init__(self, stream=sys.stdout, indent_step=2, show_ret=False):
        self.stream = stream
        self.indent_step = indent_step
        self.show_ret = show_ret

        # This is a class attribute since we want to share the indentation
        # level between different traced functions, in case they call
        # each other.
        TraceCalls.cur_indent = 0

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            indent = " " * TraceCalls.cur_indent
            argstr = ", ".join(
                [repr(a) for a in args]
                + ["%s=%s" % (a, repr(b)) for a, b in kwargs.items()]
            )
            self.stream.write("%s%s(%s)\n" % (indent, fn.__name__, argstr))

            TraceCalls.cur_indent += self.indent_step
            ret = fn(*args, **kwargs)
            TraceCalls.cur_indent -= self.indent_step

            if self.show_ret:
                self.stream.write("%s--> %s\n" % (indent, ret))
            return ret

        return wrapper
