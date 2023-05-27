from typing import Any
from gradio.utils import TraceCalls
from gradio.warnings import logger


class Dummy:
    pass
    # def __setattr__(self, __name: str, __value: Any) -> None:
    #     if hasattr(self, __name):
    #         return super().__setattr__(__name, __value)
    #     logger.info(f"Set attribute on {self.__class__} {__name} is ignored.")
    #     return None

    # def __getattribute__(self, __name: str) -> Any:
    #     if hasattr(self, __name):
    #         return super().__getattribute__(__name)
    #     logger.info(f"Get attribute on {self.__class__} {__name} is ignored.")
    #     return Dummy()

    # def __call__(self, *args: Any, **kwds: Any) -> Any:
    #     logger.info(f"Call on {self.__class__} is ignored.")
    #     return Dummy()

    # def __new__(cls, *args, **kwargs):
    #     obj = super().__new__(cls)
    #     obj.__init__(*args, **kwargs)
    #     print(f"Created {obj.__class__}")
    #     return obj
