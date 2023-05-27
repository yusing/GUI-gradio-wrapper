import json
import threading
from typing import Any, Callable, Literal

import dearpygui.dearpygui as dpg
import yaml
from fastapi import FastAPI
from typing_extensions import override

from gradio.components import CallbackGroup, Component
from gradio.dummy import Dummy
from gradio.layouts import Container
from gradio.themes import Theme
from gradio.utils import TraceCalls, logger
from gradio.warnings import warn_kwargs


class DummyApp(FastAPI):
    pass


class DummyServer:
    def __setattr__(self, __name: str, __value: Any) -> None:
        pass


class Blocks(Container):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Parameters:
            theme: a Theme object or a string representing a theme. If a string, will look for a built-in theme with that name (e.g. "soft" or "default"), or will attempt to load a theme from the HF Hub (e.g. "gradio/monochrome"). If None, will use the Default theme.
            analytics_enabled: whether to allow basic telemetry. If None, will use GRADIO_ANALYTICS_ENABLED environment variable or default to True.
            mode: a human-friendly name for the kind of Blocks or Interface being created.
            title: The tab title to display when this is opened in a browser window.
            css: custom css or path to custom css file to apply to entire Blocks
        """

        super().__init__(*args, **kwargs)
        self.load_callback = CallbackGroup()

    def render(self):
        pass

    def load(self, fn, *args, **kwargs):
        self.load_callback.append(fn)

    @override
    def build_container(self):
        return dpg.group()

    def launch(self, *args, height: int = 500, width: int | str = "100%", **kwargs):
        # logger.info(repr(self))
        try:
            Component.build()
        except Exception as e:
            with open("debug_layout.yml", "w") as f:
                f.write(
                    yaml.dump(
                        Component.current_scope.debug_info(), indent=2, sort_keys=False
                    )
                )
            # logger.error(e)
            raise e

        def run():
            dpg.create_viewport(
                title=self.user_data.get("title", "Gradio"),
                width=int(10.0 * int(width[:-1])) if isinstance(width, str) else width,
                height=height,
                small_icon=self.user_data.get("favicon_path", ""),
            )
            dpg.setup_dearpygui()
            dpg.show_viewport()
            dpg.start_dearpygui()
            dpg.destroy_context()

        threading.Thread(target=run).start()
        return DummyApp(), "GUI", "GUI"

    def queue(self, *_, **__):
        pass

    @property
    def children(self):
        return self.local_scope

    @property
    def fns(self):
        return []

    @property
    def server(self):
        return DummyServer()

    def __getattr__(self, __name: str) -> Any:
        if __name in self.__dict__:
            return self.__dict__.get(__name)
        else:
            logger.error(f'"{__name}" is not a valid attribute of this Blocks object.')
            return None


class Block(Component):
    pass


class BlockContext(Component):
    pass
