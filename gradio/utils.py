from functools import wraps
from io import BytesIO
import logging
import os
import sys
from typing import Callable, Optional
import numpy as np
import base64
import PIL.Image as _Image
import requests

logger = logging.getLogger(__name__)
EMPTY_IMG = "iVBORw0KGgoAAAANSUhEUgAAASwAAAEsCAQAAADTdEb+AAACHElEQVR42u3SMQ0AAAzDsJU/6aGo+tgQouSgIBJgLIyFscBYGAtjgbEwFsYCY2EsjAXGwlgYC4yFsTAWGAtjYSwwFsbCWGAsjIWxwFgYC2OBsTAWxgJjYSyMBcbCWBgLjIWxMBYYC2NhLDAWxsJYYCyMhbHAWBgLY4GxMBbGAmNhLIwFxsJYGAuMhbEwFhgLY2EsMBbGwlhgLIyFscBYGAtjgbEwFsYCY2EsjAXGwlgYC2OBsTAWxgJjYSyMBcbCWBgLjIWxMBYYC2NhLDAWxsJYYCyMhbHAWBgLY4GxMBbGAmNhLIwFxsJYGAuMhbEwFhgLY2EsMBbGwlhgLIyFscBYGAtjgbEwFsYCY2EsjAXGwlgYC4yFsTAWGAtjYSwwFsbCWGAsjIWxwFgYC2OBsTAWxgJjYSyMBcbCWBgLjIWxMBbGAmNhLIwFxsJYGAuMhbEwFhgLY2EsMBbGwlhgLIyFscBYGAtjgbEwFsYCY2EsjAXGwlgYC4yFsTAWGAtjYSwwFsbCWGAsjIWxwFgYC2OBsTAWxgJjYSyMBcbCWBgLjIWxMBYYC2NhLDAWxsJYYCyMhbHAWBgLY4GxMBbGAmNhLIwFxsJYGAuMhbEwFhgLY2EsjCUBxsJYGAuMhbEwFhgLY2EsMBbGwlhgLIyFscBYGAtjgbEwFsYCY2EsjAXGwlgYC4yFsTAWGAtjYSwwFsbCWGAsjIWxwFjsPeVaAS0/Qs6MAAAAAElFTkSuQmCC"


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


def to_base64_img(
    img: np.ndarray | _Image.Image | str | Callable | None,
) -> tuple[str, Optional[int], Optional[int]]:
    if callable(img):
        img = img()
    if img is None:
        return EMPTY_IMG, 300, 300
    if isinstance(img, str):  # path or base64 string
        if os.path.exists(img):
            img = _Image.open(img)
        elif img.startswith("http"):
            img = _Image.open(BytesIO(requests.get(img).content))
        else:
            assert len(img) % 4 == 0, "Base64 string is invalid."
            return img, None, None
    elif isinstance(img, np.ndarray):
        img = _Image.fromarray(img)
    byteio = BytesIO()
    img.save(byteio, format="PNG")
    img_b64 = base64.b64encode(byteio.getvalue()).decode("utf-8")
    return img_b64, img.width, img.height


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
