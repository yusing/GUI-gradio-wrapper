from typing import Optional
from typing_extensions import override
from gradio.components import Component, Container
import dearpygui.dearpygui as dpg


class Group(Container):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, excluded_attr=["callback"])

    @override
    def build_container(self, parent: Optional["Container"]):
        return dpg.group(**self.__dict__)


class Row(Group):
    @override
    def build_container(self, parent: Optional["Container"]):
        return dpg.group(**self.__dict__, horizontal=True)


class Column(Group):
    @override
    def build_container(self, parent: Optional["Container"]):
        return dpg.group(**self.__dict__, width=self.user_data.get("min_width", 0))


class Tabs(Container):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, excluded_attr=["drag_callback"])

    @override
    def build_container(self, parent: Optional["Container"]):
        return dpg.tab_bar(**self.__dict__)

    @property
    def selected(self):
        return dpg.get_value(self.tag)


class Tab(Container):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, excluded_attr=["callback", "drag_callback"])

    @override
    def build_container(self, parent: Optional["Container"]):
        if not isinstance(parent, Tabs):
            with dpg.tab_bar():
                return dpg.tab(**self.__dict__)
        return dpg.tab(**self.__dict__)


class Accordion(Container):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, excluded_attr=["callback"])

    @override
    def build_container(self, parent: Optional["Container"]):
        return dpg.collapsing_header(
            **self.__dict__,
            default_open=self.user_data.get("open", True),
            open_on_arrow=True,
            open_on_double_click=True,
        )


class Box(Group):
    pass


TabItem = Tab
