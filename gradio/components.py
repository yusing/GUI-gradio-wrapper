import json
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from inspect import stack
from types import NoneType
from typing import Any, Callable, ContextManager, Optional, Sequence, Union

import flet as ft
import numpy as np
from PIL import Image as _Image
from typing_extensions import override

from gradio.helpers import update as _update
from gradio.utils import hex_to_rgba, logger, to_base64_img
from gradio.warnings import warn_unimplemented

CallbackIO = Union["Component", list["Component"], set["Component"], None]


class CallbackGroup:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[], None]] = []

    def __call__(self, _=None) -> list[Any]:
        return [fn() for fn in self.callbacks]

    def __add__(self, other: "CallbackGroup") -> "CallbackGroup":
        new = CallbackGroup()
        new.callbacks.extend(self.callbacks)
        new.callbacks.extend(other.callbacks)
        return new

    def append(self, fn: Optional[Callable]):
        if fn is None:
            return
        self.callbacks.append(fn)


def event_handler(event_name: str):
    class GradioEventHandler:
        def __init__(self, *args, **kwargs) -> None:
            # print(
            #     f"Event handler {event_name} initialized for {self.__class__.__name__}"
            # )
            if f"on_{event_name}" not in self.__dict__:
                setattr(self, f"on_{event_name}", CallbackGroup())
            if event_name not in self.__dict__:
                setattr(
                    self,
                    event_name,
                    lambda fn, inputs, outputs, *_, **__: Component.add_callback(
                        getattr(self, f"on_{event_name}"), fn, inputs, outputs
                    ),
                )

    return GradioEventHandler


class Component(ABC):
    component_map: dict[str | int, "Component"] = {}

    def init_event_handlers(self, classes):
        for baseclass in classes:
            if baseclass.__name__ == "GradioEventHandler":
                baseclass.__init__(self)  # type: ignore
            self.init_event_handlers(baseclass.__bases__)

    def __init__(self, *args, append_to_scope=True, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.traceback = stack(1)[2]
        self.element: ft.Control
        self.value_thread = threading.Thread(target=self.value_callback)
        self.elem_id: Optional[str] = None
        self.parent = Container.current_scope
        self.every = 1.0
        self.label: Optional[str] = None
        self.init_event_handlers(self.__class__.__bases__)
        if append_to_scope:
            self.append_to_scope()

    def append_to_scope(self):
        from gradio.layouts import Tabs, Tab

        scope = Container.current_scope
        while isinstance(Container.current_scope, Tabs) and not isinstance(self, Tab):
            Container.current_scope = Container.current_scope.parent
        Container.current_scope.local_scope.append(self)
        if self.elem_id in Component.component_map:
            self.elem_id = f"{self.elem_id}_{len(Component.component_map)}"
        Component.component_map[self.elem_id] = self
        Container.current_scope = scope

    @staticmethod
    def dynamic(t: Union[str, "Component"]) -> "Component":
        if isinstance(t, Component):
            return t
        for cls in Component.__subclasses__():
            if cls.__name__.lower() == t.lower():
                return cls()  # type: ignore
        assert False, f"Component type {t} not found"

    @staticmethod
    def add_callback(
        callback: CallbackGroup,
        fn: Callable,
        inputs: CallbackIO,
        outputs: CallbackIO,
    ):
        if fn is None:
            return
        if not isinstance(inputs, (list, set)):
            inputs = [inputs] if inputs else []
        if not isinstance(outputs, (list, set)):
            outputs = [outputs] if outputs else []

        def handle_callback(
            results: Sequence[Any] | Any | dict["Component", Any | dict[str, Any]]
        ):
            if isinstance(results, (list, set)):
                if len(results) == len(outputs):  # type: ignore
                    for result, output in zip(results, outputs):  # type: ignore
                        output.set_value(result)
                else:
                    for output in outputs:  # type: ignore
                        output.set_value(results)
            elif isinstance(results, dict):
                for output, value in results.items():
                    if isinstance(output, str):
                        continue  # TODO: Handle this
                    if isinstance(value, dict):
                        for k, v in value.items():
                            setattr(output.element, k, v)
                        output.element.update()
                    else:
                        output.set_value(value)
            else:
                for output in outputs:  # type: ignore
                    output.set_value(results)

        def run_callback():
            # print("Inputs:", inputs)
            # print("Outputs:", outputs)
            if inputs is None:
                results = fn()
            else:
                results = fn(*[input_.get_value() for input_ in inputs if input_])
            if isinstance(results, ContextManager):
                with results as results:
                    handle_callback(results)
            else:
                handle_callback(results)

        callback.append(run_callback)

    @abstractmethod
    def build(self) -> ft.Control:
        raise NotImplementedError

    def get_value(self):
        return self.element.value

    def set_value(self, value):
        if callable(value):
            self.value = value
            return
        self.element.value = value
        self.element.update()

    def value_callback(self):
        while True:
            if callable(self.value):
                self.set_value(self.value())
            elif self.value and self.get_value() is None:
                self.set_value(self.value)
            time.sleep(self.every)

    def update(**kwargs):
        return _update(**kwargs)

    def update_element(self):
        self.element.update()

    @warn_unimplemented
    def style(self, *args, **kwargs):
        return self

    @property
    def attributes(self) -> dict[str, Any]:
        return self.__dict__

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"<{self.__class__.__name__} elem_id={self.elem_id} caller={os.path.basename(self.traceback.filename)}:{self.traceback.lineno}>"

    def debug_info(self):
        info = {
            "element": repr(self),
            # "parent": repr(self.parent),
        }
        if isinstance(self, Container):
            info["children"] = [ele.debug_info() for ele in self.local_scope]
        return info


class Container(Component):
    current_scope: "Container"

    def __init__(self, *args, **kwargs) -> None:
        self.local_scope: list["Component"] = []
        super().__init__(*args, **kwargs)

    def __enter__(self):
        Container.current_scope = self
        return self

    def __exit__(self, *_):
        Container.current_scope = self.parent
        assert Container.current_scope is not None, "No previous scope to exit to"

    def build_children(self):
        from gradio.layouts import Tabs, Tab

        if not isinstance(self, Tabs) and any(
            map(lambda e: isinstance(e, Tab), self.local_scope)
        ):
            tabs = []
            i = 0
            while i < len(self.local_scope):
                if isinstance(self.local_scope[i], Tab):
                    tab = self.local_scope.pop(i)
                    tabs.append(tab)
                else:
                    i += 1

            tabs_element = Tabs(append_to_scope=False)
            tabs_element.local_scope = tabs
            tabs_element.parent = self
            for t in tabs:
                t.parent = tabs_element
            self.local_scope.append(tabs_element)
        return [e.build() for e in self.local_scope]

    @property
    def children(self):
        return self.local_scope

    @staticmethod
    def init_scope():
        assert not hasattr(Container, "current_scope")
        from gradio.layouts import Column

        Container.current_scope = None
        Container.current_scope = Column(expand=True, append_to_scope=False)


class Changeable(event_handler("change")):  # type: ignore
    pass


class Clearable(event_handler("clear")):  # type: ignore
    pass


class Editable(event_handler("edit")):  # type: ignore
    pass


class Clickable(event_handler("click")):  # type: ignore
    pass


class Submittable(event_handler("submit")):  # type: ignore
    pass


class Selectable(event_handler("select")):  # type: ignore
    pass


class Inputable(event_handler("input")):  # type: ignore
    pass


class Blurrable(event_handler("blur")):  # type: ignore
    pass


class Releaseable(event_handler("release")):  # type: ignore
    pass


class Streamable(event_handler("stream")):  # type: ignore
    pass


class Uploadable(event_handler("upload")):  # type: ignore
    pass


class Playable(event_handler("play")):  # type: ignore
    pass


class IOComponent(Component):
    pass


class TextBox(Component, Changeable, Inputable, Selectable, Submittable, Blurrable):
    """
    Creates a textarea for user to enter string input or display string output.
    Preprocessing: passes textarea value as a {str} into the function.
    Postprocessing: expects a {str} returned from function and sets textarea value to it.
    Examples-format: a {str} representing the textbox input.

    Demos: hello_world, diff_texts, sentence_builder
    Guides: creating-a-chatbot, real-time-speech-recognition
    """

    def __init__(
        self,
        value: str | Callable | None = "",
        *,
        lines: int = 1,
        max_lines: int = 20,
        placeholder: str | None = None,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        type: str = "text",
        **kwargs,
    ):
        """
        Parameters:
            value: default text to provide in textarea. If callable, the function will be called whenever the app loads to set the initial value of the component.
            lines: minimum number of line rows to provide in textarea.
            max_lines: maximum number of line rows to provide in textarea.
            placeholder: placeholder hint to provide behind textarea.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will be rendered as an editable textbox; if False, editing will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            type: The type of textbox. One of: 'text', 'password', 'email', Default is 'text'.
        """
        super().__init__(**kwargs)
        if type not in ["text", "password", "email"]:
            raise ValueError('`type` must be one of "text", "password", or "email".')
        self.value = value
        self.lines = lines
        if type == "text":
            self.max_lines = max(lines, max_lines)
        else:
            self.max_lines = 1
        self.placeholder = placeholder
        self.label = label
        self.info = info
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.type = type
        self.element: ft.TextField = None

    @override
    def build(self):
        self.element = ft.TextField(
            key=self.elem_id,
            value=self.value,
            min_lines=self.lines,
            max_lines=self.max_lines,
            hint_text=self.placeholder,
            label=self.label,
            label_style=None if self.show_label else ft.TextStyle(0),
            helper_text=self.info,
            read_only=not self.interactive,
            password=self.type == "password",
            visible=self.visible,
            keyboard_type=ft.KeyboardType.EMAIL if self.type == "email" else None,
            on_change=self.on_change + self.on_input,
            on_submit=self.on_submit,
            on_blur=self.on_blur,
        )
        self.value_thread.start()
        return self.element


class Number(
    Component,
    Changeable,
    Inputable,
    Submittable,
    Blurrable,
):
    """
    Creates a numeric field for user to enter numbers as input or display numeric output.
    Preprocessing: passes field value as a {float} or {int} into the function, depending on `precision`.
    Postprocessing: expects an {int} or {float} returned from the function and sets field value to it.
    Examples-format: a {float} or {int} representing the number's value.

    Demos: tax_calculator, titanic_survival, blocks_simple_squares
    """

    def __init__(
        self,
        value: float | Callable | None = None,
        *,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        precision: int | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: default value. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will be editable; if False, editing will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            precision: Precision to round input/output to. If set to 0, will round to nearest integer and convert type to int. If None, no rounding happens.
        """
        super().__init__(**kwargs)
        self.value = value
        self.label = label
        self.info = info
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.precision = precision
        self.element: ft.TextField = None

    @override
    def build(self):
        self.element = ft.TextField(
            key=self.elem_id,
            label=self.label,
            label_style=None if self.show_label else ft.TextStyle(0),
            helper_text=self.info,
            visible=self.visible,
            read_only=not self.interactive,
            on_change=self.on_change + self.on_input,
            on_submit=self.on_submit,
            on_blur=self.on_blur,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        self.value_thread.start()
        return self.element

    @override
    def get_value(self):
        if self.precision is None:
            return int(self.element.value) if self.element.value else 0
        return float(self.element.value) if self.element.value else 0.0

    @override
    def set_value(self, value: int | float | Callable):
        if callable(value):
            self.value = value
            return
        assert isinstance(value, (int, float))
        if self.precision is None:
            value = float(value)
        elif self.precision == 0:
            value = int(round(value))
        else:
            value = round(value, self.precision)
        return super().set_value(f"{value}")


class Image(
    Component, Editable, Clearable, Changeable, Streamable, Selectable, Uploadable
):
    """
    Creates an image component that can be used to upload/draw images (as an input) or display images (as an output).
    Preprocessing: passes the uploaded image as a {numpy.array}, {PIL.Image} or {str} filepath depending on `type` -- unless `tool` is `sketch` AND source is one of `upload` or `webcam`. In these cases, a {dict} with keys `image` and `mask` is passed, and the format of the corresponding values depends on `type`.
    Postprocessing: expects a {numpy.array}, {PIL.Image} or {str} or {pathlib.Path} filepath to an image and displays the image.
    Examples-format: a {str} filepath to a local file that contains the image.
    Demos: image_mod, image_mod_default_image
    Guides: image-classification-in-pytorch, image-classification-in-tensorflow, image-classification-with-vision-transformers, building-a-pictionary_app, create-your-own-friends-with-a-gan
    """

    def __init__(
        self,
        value: str | _Image.Image | np.ndarray | None = None,
        *,
        shape: tuple[int, int] | None = None,
        image_mode: str = "RGB",
        invert_colors: bool = False,
        source: str = "upload",
        tool: str | None = None,
        type: str = "numpy",
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        mirror_webcam: bool = True,
        brush_radius: float | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: A PIL Image, numpy array, path or URL for the default value that Image component is going to take. If callable, the function will be called whenever the app loads to set the initial value of the component.
            shape: (width, height) shape to crop and resize image to; if None, matches input image size. Pass None for either width or height to only crop and resize the other.
            image_mode: "RGB" if color, or "L" if black and white.
            invert_colors: whether to invert the image as a preprocessing step.
            source: Source of image. "upload" creates a box where user can drop an image file, "webcam" allows user to take snapshot from their webcam, "canvas" defaults to a white image that can be edited and drawn upon with tools.
            tool: Tools used for editing. "editor" allows a full screen editor (and is the default if source is "upload" or "webcam"), "select" provides a cropping and zoom tool, "sketch" allows you to create a binary sketch (and is the default if source="canvas"), and "color-sketch" allows you to created a sketch in different colors. "color-sketch" can be used with source="upload" or "webcam" to allow sketching on an image. "sketch" can also be used with "upload" or "webcam" to create a mask over an image and in that case both the image and mask are passed into the function as a dictionary with keys "image" and "mask" respectively.
            type: The format the image is converted to before being passed into the prediction function. "numpy" converts the image to a numpy array with shape (height, width, 3) and values from 0 to 255, "pil" converts the image to a PIL image object, "filepath" passes a str path to a temporary file containing the image.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will allow users to upload and edit an image; if False, can only be used to display images. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            streaming: If True when used in a `live` interface, will automatically stream webcam feed. Only valid is source is 'webcam'.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            mirror_webcam: If True webcam will be mirrored. Default is True.
            brush_radius: Size of the brush for Sketch. Default is None which chooses a sensible default
        """
        super().__init__(**kwargs)
        self.value = value
        self.shape = shape
        self.image_mode = image_mode  #! ignored
        self.invert_colors = invert_colors  #! ignored
        self.source = source  #! ignored
        self.type = type  #! ignored
        self.label = label
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True  #! ignored
        self.visible = visible
        self.streaming = streaming
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.mirror_webcam = mirror_webcam  #! ignored
        self.brush_radius = brush_radius  #! ignored
        self.element: ft.Column = None

        assert self.source in [
            "upload",
            "webcam",
            "canvas",
        ], f"Invalid source: {source}"
        assert self.type in ["numpy", "pil", "filepath"], f"Invalid type: {type}"
        if tool is None:
            self.tool = "sketch" if source == "canvas" else "editor"
        else:
            self.tool = tool
        if streaming and source != "webcam":
            raise ValueError("Image streaming only available if source is 'webcam'.")

    @override
    def build(self):
        self.change_callback = (
            self.on_change + self.on_edit + self.on_stream + self.on_upload
        )
        img, width, height = to_base64_img(self.value)
        self.image = ft.Image(
            src_base64=img,
            width=width,
            height=height,
            key=self.elem_id,
            visible=self.visible,
        )
        self.element = ft.Stack(
            [
                self.image,
                ft.Container(
                    ft.Text(self.label),
                    bgcolor="#666666",
                    opacity=0.5,
                    visible=self.show_label,
                    padding=ft.padding.symmetric(horizontal=4, vertical=0),
                ),
            ],
        )
        self.value_thread.start()
        return self.element

    @override
    def set_value(self, value: np.ndarray | _Image.Image | str | Callable | None):
        if callable(value):
            self.value = value
            return
        assert isinstance(value, (np.ndarray, _Image.Image, str, NoneType))
        (
            self.image.src_base64,
            self.image.width,
            self.image.height,
        ) = to_base64_img(value)
        self.parent.element.update()
        self.change_callback()

    @override
    def get_value(self) -> Optional[str]:
        return self.image.src_base64


class CheckboxGroup(Component, Changeable, Inputable, Selectable):
    """
    Creates a set of checkboxes of which a subset can be checked.
    Preprocessing: passes the list of checked checkboxes as a {List[str]} or their indices as a {List[int]} into the function, depending on `type`.
    Postprocessing: expects a {List[str]}, each element of which becomes a checked checkbox.
    Examples-format: a {List[str]} representing the values to be checked.
    Demos: sentence_builder, titanic_survival
    """

    def __init__(
        self,
        choices: list[str] | None = None,
        *,
        value: list[str] | str | Callable | None = None,
        type: str = "value",
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            choices: list of options to select from.
            value: default selected list of options. If callable, the function will be called whenever the app loads to set the initial value of the component.
            type: Type of value to be returned by component. "value" returns the list of strings of the choices selected, "index" returns the list of indices of the choices selected.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, choices in this checkbox group will be checkable; if False, checking will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs)
        self.choices = choices or []
        self.value = value or []
        self.type = type
        self.label = label
        self.info = info
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.element: ft.Column = None
        assert self.type in ["value", "index"], f"Invalid type: {type}"
        if isinstance(self.value, str):
            self.value = [self.value]

    @override
    def build(self):
        self.checkboxes = [
            ft.Checkbox(
                label=choice,
                value=choice in self.value,
                on_change=self.on_change + self.on_input,
            )
            for choice in self.choices
        ]
        self.element = ft.Column(
            [
                ft.Text(self.label, visible=self.show_label, tooltip=self.info),
                ft.Row(self.checkboxes, spacing=8),
            ],
            key=self.elem_id,
            spacing=8,
            visible=self.visible,
            disabled=not self.interactive,
        )
        self.value_thread.start()
        return self.element

    @override
    def get_value(self):
        if self.type == "value":
            return [cb.label for cb in self.checkboxes if cb.value]
        return [i for i, cb in enumerate(self.checkboxes) if cb.value]

    @override
    def set_value(self, values: list[str | int] | Callable):
        if callable(values):
            self.value = values
            return
        assert isinstance(values, list)
        if not any(values):
            values = []
        if self.type == "value":
            for cb in self.checkboxes:
                cb.value = cb.label in values
        else:
            for i, cb in enumerate(self.checkboxes):
                cb.value = i in values
        self.element.update()


class Radio(Component, Selectable, Changeable, Inputable):
    """
    Creates a set of radio buttons of which only one can be selected.
    Preprocessing: passes the value of the selected radio button as a {str} or its index as an {int} into the function, depending on `type`.
    Postprocessing: expects a {str} corresponding to the value of the radio button to be selected.
    Examples-format: a {str} representing the radio option to select.

    Demos: sentence_builder, titanic_survival, blocks_essay
    """

    def __init__(
        self,
        choices: list[str] | None = None,
        *,
        value: str | Callable | None = None,
        type: str = "value",
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            choices: list of options to select from.
            value: the button selected by default. If None, no button is selected by default. If callable, the function will be called whenever the app loads to set the initial value of the component.
            type: Type of value to be returned by component. "value" returns the string of the choice selected, "index" returns the index of the choice selected.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, choices in this radio group will be selectable; if False, selection will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs)
        self.choices = choices or []
        self.value = value
        self.type = type
        self.label = label
        self.info = info
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.element: ft.Column = None

        assert self.type in ["value", "index"], f"Invalid type: {type}"

    @override
    def build(self):
        self.radio_buttons = [
            ft.Radio(label=choice, value=choice) for choice in self.choices
        ]
        self.radio_group = ft.RadioGroup(
            ft.Column(self.radio_buttons),
            value=self.value,
            on_change=self.on_change + self.on_input,
        )
        self.element = ft.Column(
            [
                ft.Text(self.label, visible=self.show_label, tooltip=self.info),
                self.radio_group,
            ],
            key=self.elem_id,
            visible=self.visible,
            disabled=not self.interactive,
        )
        self.value_thread.start()
        return self.element

    @override
    def get_value(self):
        if self.type == "value":
            return self.radio_group.value
        return self.choices.index(self.radio_group.value)

    @override
    def set_value(self, value: str | int | Callable | None):
        if callable(value):
            self.value = value
            return
        if value is None:
            self.radio_group.value = None
            self.radio_group.update()
            return
        if self.type == "value":
            assert isinstance(value, str), f"Invalid value: {value}"
            self.radio_group.value = value
        else:
            assert isinstance(value, int), f"Invalid value: {value}"
            self.radio_group.value = self.choices[value]
        self.radio_group.update()


class ColorPicker(Component, Changeable, Inputable, Submittable, Blurrable):
    """
    Creates a color picker for user to select a color as string input.
    Preprocessing: passes selected color value as a {str} into the function.
    Postprocessing: expects a {str} returned from function and sets color picker value to it.
    Examples-format: a {str} with a hexadecimal representation of a color, e.g. "#ff0000" for red.
    Demos: color_picker, color_generator
    """

    def __init__(
        self,
        value: str | Callable | None = None,
        *,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: default text to provide in color picker. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will be rendered as an editable color picker; if False, editing will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(self, **kwargs)
        self.value = value or "#FFFFFF"
        self.label = label or "Color"
        self.info = info
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.element: ft.Column
        self.set_color = self.on_change + self.on_input
        self.set_color.append(self.update_color)

    def slider(self, color: str, value: int):
        return (
            ft.Text(color),
            ft.Slider(
                min=0,
                max=255,
                divisions=255,
                round=0,
                value=value,
                on_change=self.set_color,
                on_change_end=self.on_submit,
                on_blur=self.on_blur,
            ),
            ft.Text(str(value)),
        )

    def update_slider(self, slider_row: tuple[ft.Text, ft.Slider, ft.Text], value: int):
        _, slider, value_label = slider_row
        slider.value = value
        slider.update()
        value_label.value = str(value)
        value_label.update()

    def update_color(self, color: Optional[str] = None):
        color = color or self.get_value()
        r, g, b, _ = hex_to_rgba(color)
        self.update_slider(self.red_slider, r)
        self.update_slider(self.green_slider, g)
        self.update_slider(self.blue_slider, b)
        self.color.controls[0].bgcolor = color
        self.color.controls[1].value = color
        self.color.update()

    @override
    def build(self):
        r, g, b, _ = hex_to_rgba(self.value)
        self.red_slider = self.slider("R", r)
        self.green_slider = self.slider("G", g)
        self.blue_slider = self.slider("B", b)
        self.color = ft.Row(
            [
                ft.Container(
                    width=32,
                    height=32,
                    bgcolor=self.value,
                    shape=ft.BoxShape.CIRCLE,
                ),
                ft.Text(self.value),
            ],
            tight=True,
            wrap=False,
        )
        self.element = ft.Column(
            [
                ft.ListTile(
                    title=ft.Text(self.label, visible=self.show_label),
                    trailing=self.color,
                    subtitle=ft.Text(self.info) if self.info else None,
                    content_padding=ft.padding.all(0),
                ),
                ft.Row(
                    [
                        ft.Row(self.red_slider),
                        ft.Container(width=16),
                        ft.Row(self.green_slider),
                        ft.Container(width=16),
                        ft.Row(self.blue_slider),
                    ],
                    wrap=True,
                    tight=True,
                ),
            ],
            visible=self.visible,
            disabled=not self.interactive,
            spacing=8,
            key=self.elem_id,
            tight=True,
            wrap=False,
        )
        self.value_thread.start()
        return self.element

    @override
    def get_value(self):
        red = hex(int(self.red_slider[1].value))[2:].zfill(2)
        green = hex(int(self.green_slider[1].value))[2:].zfill(2)
        blue = hex(int(self.blue_slider[1].value))[2:].zfill(2)
        return f"#{red}{green}{blue}".upper()

    @override
    def set_value(self, value: str | Callable | None):
        if callable(value):
            self.value = value
            return
        value = value or "#FFFFFF"
        assert isinstance(value, str)
        assert len(value) == 7 and value.startswith("#"), f"Invalid value: {value}"
        self.update_color(value)
        # self.element.update()


class DropDown(Component, Changeable, Inputable, Selectable, Blurrable):
    """
    Creates a dropdown of choices from which entries can be selected.
    Preprocessing: passes the value of the selected dropdown entry as a {str} or its index as an {int} into the function, depending on `type`.
    Postprocessing: expects a {str} corresponding to the value of the dropdown entry to be selected.
    Examples-format: a {str} representing the drop down value to select.
    Demos: sentence_builder, titanic_survival
    """

    def __init__(
        self,
        choices: list[str] | None = None,
        *,
        value: str | list[str] | Callable | None = None,
        type: str = "value",
        multiselect: bool | None = None,
        max_choices: int | None = None,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        allow_custom_value: bool = False,
        **kwargs,
    ):
        """
        Parameters:
            choices: list of options to select from.
            value: default value(s) selected in dropdown. If None, no value is selected by default. If callable, the function will be called whenever the app loads to set the initial value of the component.
            type: Type of value to be returned by component. "value" returns the string of the choice selected, "index" returns the index of the choice selected.
            multiselect: if True, multiple choices can be selected.
            max_choices: maximum number of choices that can be selected. If None, no limit is enforced.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, choices in this dropdown will be selectable; if False, selection will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            allow_custom_value: If True, allows user to enter a custom value that is not in the list of choices.
        """
        super().__init__(**kwargs)
        self.choices = choices or []
        self.value = value
        self.type = type
        self.multiselect = multiselect  #! ignored
        self.max_choices = max_choices  #! ignored
        self.label = label
        self.info = info
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.allow_custom_value = allow_custom_value  #! ignored
        self.element: ft.Dropdown

        assert self.type in ["value", "index"], f"Invalid type: {type}"

    def build_dropdown_items(self, items: list[str]):
        return list(map(ft.dropdown.Option, items))

    @override
    def build(self):
        self.element = ft.Dropdown(
            value=self.value,
            tooltip=self.info,
            key=self.elem_id,
            visible=self.visible,
            disabled=not self.interactive,
            label=self.label,
            label_style=None if self.show_label else ft.TextStyle(0),
            options=self.build_dropdown_items(self.choices),
            on_change=self.on_change + self.on_input,
            on_blur=self.on_blur,
            on_focus=self.on_select,
        )
        self.value_thread.start()
        return self.element

    @override
    def get_value(self):
        if self.type == "value":
            return self.element.value
        return self.choices.index(self.element.value)

    @override
    def set_value(self, value: str | int | Callable):
        if callable(value):
            self.value = value
            return

        if self.type == "value":
            assert isinstance(value, str), f"Invalid value: {value}"
            self.element.value = value
        else:
            assert isinstance(value, int), f"Invalid value: {value}"
            self.element.value = self.choices[value]
        self.element.update()


class Label(Component, Changeable, Selectable):
    """
    Displays a classification label, along with confidence scores of top categories, if provided.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a {Dict[str, float]} of classes and confidences, or {str} with just the class or an {int}/{float} for regression outputs, or a {str} path to a .json file containing a json dictionary in the structure produced by Label.postprocess().

    Demos: main_note, titanic_survival
    Guides: image-classification-in-pytorch, image-classification-in-tensorflow, image-classification-with-vision-transformers, building-a-pictionary-app
    """

    def __init__(
        self,
        value: dict[str, float] | str | float | Callable | None = None,
        *,
        num_top_classes: int | None = None,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        color: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value = value
        self.num_top_classes = num_top_classes
        self.label = label
        self.every = every if every else 1
        self.show_label = show_label
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.color = color
        self.element: ft.Row

    @override
    def build(self):
        self.element = ft.Row(
            [
                ft.Text(self.label, visible=self.show_label),
                ft.Text(self.value, color=self.color),
            ],
            key=self.elem_id,
            visible=self.visible,
            spacing=8,
        )
        self.value_thread.start()
        return self.element

    @override
    def get_value(self):
        return self.element.controls[1].value

    @override
    def set_value(self, value: dict[str, float] | str | float | Callable):
        if callable(value):
            self.value = value
            return

        if isinstance(value, dict):
            self.element.controls[1].value = json.dumps(value, indent=2)
        elif isinstance(value, (str, float, int)):
            self.element.controls[1].value = str(value)
        else:
            raise ValueError(f"Invalid value: {value}")
        self.element.update()


class File(Component, Changeable, Selectable, Clearable, Uploadable):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def build(self):
        self.element = ft.Container()  # TODO: fix this
        return self.element


class Slider(Component, Changeable, Inputable, Releaseable):
    """
    Creates a slider that ranges from `minimum` to `maximum` with a step size of `step`.
    Preprocessing: passes slider value as a {float} into the function.
    Postprocessing: expects an {int} or {float} returned from function and sets slider value to it as long as it is within range.
    Examples-format: A {float} or {int} representing the slider's value.

    Demos: sentence_builder, slider_release, generate_tone, titanic_survival, interface_random_slider, blocks_random_slider
    Guides: create-your-own-friends-with-a-gan
    """

    def __init__(
        self,
        minimum: float = 0,
        maximum: float = 100,
        value: float | Callable | None = None,
        *,
        step: float | None = None,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        randomize: bool = False,
        **kwargs,
    ):
        """
        Parameters:
            minimum: minimum value for slider.
            maximum: maximum value for slider.
            value: default value. If callable, the function will be called whenever the app loads to set the initial value of the component. Ignored if randomized=True.
            step: increment between slider values.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, slider will be adjustable; if False, adjusting will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            randomize: If True, the value of the slider when the app loads is taken uniformly at random from the range given by the minimum and maximum.
        """
        super().__init__(**kwargs)
        self.minimum = minimum
        self.maximum = maximum
        self.value = value
        self.step = step if step else 1
        self.label = label
        self.info = info
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.randomize = randomize
        self.element: ft.Slider

    @override
    def build(self):
        self.element = ft.Slider(
            tooltip=self.info,
            key=self.elem_id,
            visible=self.visible,
            disabled=not self.interactive,
            label=self.label if self.show_label else None,
            divisions=(self.maximum - self.minimum) // self.step,
            min=self.minimum,
            max=self.maximum,
            on_change=self.on_change + self.on_input,
            on_change_end=self.on_release,
            value=self.value
            if not self.randomize
            else random.uniform(self.minimum, self.maximum),
        )
        self.value_thread.start()
        return self.element


class Button(Component, Clickable):
    """
    Used to create a button, that can be assigned arbitrary click() events. The label (value) of the button can be used as an input or set via the output of a function.

    Preprocessing: passes the button value as a {str} into the function
    Postprocessing: expects a {str} to be returned from a function, which is set as the label of the button
    Demos: blocks_inputs, blocks_kinematics
    """

    def __init__(
        self,
        value: str | Callable = "Run",
        *,
        variant: str = "secondary",
        visible: bool = True,
        interactive: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default text for the button to display. If callable, the function will be called whenever the app loads to set the initial value of the component.
            variant: 'primary' for main call-to-action, 'secondary' for a more subdued style, 'stop' for a stop button.
            visible: If False, component will be hidden.
            interactive: If False, the Button will be in a disabled state.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs)
        self.value = value
        self.variant = variant
        self.visible = visible
        self.interactive = interactive or True
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.element: ft.ElevatedButton

        assert self.variant in [
            "primary",
            "secondary",
            "stop",
        ], f"Invalid variant: {variant}"

        if self.variant == "primary":
            self.bgcolor = "#FFFFFF"
        elif self.variant == "secondary":
            self.bgcolor = "#F0F0F0"
        else:
            self.bgcolor = "#FF0000"

    @override
    def build(self):
        self.element = ft.ElevatedButton(
            key=self.elem_id,
            visible=self.visible,
            disabled=not self.interactive,
            text=self.value,
            on_click=self.on_click,
            bgcolor=self.bgcolor,
        )
        self.value_thread.start()
        return self.element

    @override
    def get_value(self):
        return self.element.text

    @override
    def set_value(self, value: str | Callable):
        if callable(value):
            self.value = value
            return
        assert isinstance(value, str), f"Invalid value: {value}"
        self.element.text = value
        self.element.update()


class CheckBox(Component, Changeable, Inputable, Selectable):
    """
    Creates a checkbox that can be set to `True` or `False`.

    Preprocessing: passes the status of the checkbox as a {bool} into the function.
    Postprocessing: expects a {bool} returned from the function and, if it is True, checks the checkbox.
    Examples-format: a {bool} representing whether the box is checked.
    Demos: sentence_builder, titanic_survival
    """

    def __init__(
        self,
        value: bool | Callable = False,
        *,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: if True, checked by default. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, this checkbox can be checked; if False, checking will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """

        super().__init__(**kwargs)
        self.value = value
        self.label = label
        self.info = info
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.element: ft.Checkbox

    @override
    def build(self):
        self.element = ft.Checkbox(
            key=self.elem_id,
            visible=self.visible,
            disabled=not self.interactive,
            label=self.label if self.show_label else None,
            tooltip=self.info,
            value=self.value,
            on_change=self.on_change + self.on_input + self.on_select,
        )
        self.value_thread.start()
        return self.element


class Gallery(Component, Selectable):
    """
    Used to display a list of images as a gallery that can be scrolled through.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a list of images in any format, {List[numpy.array | PIL.Image | str]}, or a {List} of (image, {str} caption) tuples and displays them.

    Demos: fake_gan
    """

    def __init__(
        self,
        value: list[np.ndarray | _Image.Image | str | tuple] | Callable | None = None,
        *,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: List of images to display in the gallery by default. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs)
        self.value = value or []
        self.label = label
        self.every = every if every else 1
        self.show_label = show_label
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.element: ft.Column

    @override
    def build(self):
        # self.grid = ft.GridView(
        #     [
        #         ft.GestureDetector(
        #             ft.Image(
        #                 src_base64=to_base64_img(image),
        #             ),
        #             on_tap=self.on_select,
        #         )
        #         for image in self.value
        #     ]
        # )
        self.grid = ft.GridView(spacing=8)
        self.element = ft.Column(
            [ft.Text(self.label, visible=self.show_label), self.grid],
            visible=self.visible,
            key=self.elem_id,
            spacing=8,
        )
        self.value_thread.start()
        return self.element

    @override
    def get_value(self):
        return self.grid.controls

    @override
    def set_value(
        self, value: list[np.ndarray | _Image.Image | str | tuple] | Callable
    ):
        if callable(value):
            self.value = value
            return

        self.grid.controls = [
            ft.GestureDetector(
                ft.Image(
                    src_base64=to_base64_img(image),
                ),
                on_tap=self.on_select,
            )
            for image in value
        ]
        self.grid.update()


class DataFrame(Component, Changeable, Inputable, Selectable):
    """
    Accepts or displays 2D input through a spreadsheet-like component for dataframes.
    Preprocessing: passes the uploaded spreadsheet data as a {pandas.DataFrame}, {numpy.array}, {List[List]}, or {List} depending on `type`
    Postprocessing: expects a {pandas.DataFrame}, {numpy.array}, {List[List]}, {List}, a {Dict} with keys `data` (and optionally `headers`), or {str} path to a csv, which is rendered in the spreadsheet.
    Examples-format: a {str} filepath to a csv with data, a pandas dataframe, or a list of lists (excluding headers) where each sublist is a row of data.
    Demos: filter_records, matrix_transpose, tax_calculator
    """

    def __init__(
        self,
        value: list[list[Any]] | Callable | None = None,
        *,
        headers: list[str] | None = None,
        row_count: int | tuple[int, str] = (1, "dynamic"),
        col_count: int | tuple[int, str] | None = None,
        datatype: str | list[str] = "str",
        type: str = "pandas",
        max_rows: int | None = 20,
        max_cols: int | None = None,
        overflow_row_behaviour: str = "paginate",
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        wrap: bool = False,
        **kwargs,
    ):
        """
        Parameters:
            value: Default value as a 2-dimensional list of values. If callable, the function will be called whenever the app loads to set the initial value of the component.
            headers: List of str header names. If None, no headers are shown.
            row_count: Limit number of rows for input and decide whether user can create new rows. The first element of the tuple is an `int`, the row count; the second should be 'fixed' or 'dynamic', the new row behaviour. If an `int` is passed the rows default to 'dynamic'
            col_count: Limit number of columns for input and decide whether user can create new columns. The first element of the tuple is an `int`, the number of columns; the second should be 'fixed' or 'dynamic', the new column behaviour. If an `int` is passed the columns default to 'dynamic'
            datatype: Datatype of values in sheet. Can be provided per column as a list of strings, or for the entire sheet as a single string. Valid datatypes are "str", "number", "bool", "date", and "markdown".
            type: Type of value to be returned by component. "pandas" for pandas dataframe, "numpy" for numpy array, or "array" for a Python array.
            label: component name in interface.
            max_rows: Maximum number of rows to display at once. Set to None for infinite.
            max_cols: Maximum number of columns to display at once. Set to None for infinite.
            overflow_row_behaviour: If set to "paginate", will create pages for overflow rows. If set to "show_ends", will show initial and final rows and truncate middle rows.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will allow users to edit the dataframe; if False, can only be used to display data. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            wrap: if True text in table cells will wrap when appropriate, if False the table will scroll horizontally. Defaults to False.
        """
        super().__init__(**kwargs)
        self.value = value or []
        assert headers or col_count, "Must provide headers or col_count"
        self.headers = headers or list(
            map(
                lambda i: f"Column {i+1}",
                range(col_count if isinstance(col_count, int) else col_count[0]),
            )
        )
        self.row_count = row_count
        self.col_count = col_count
        self.datatype = datatype
        self.type = type
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.overflow_row_behaviour = overflow_row_behaviour
        self.label = label
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.wrap = wrap
        self.element: ft.Column
        self.columns: ft.DataColumn
        self.rows: ft.DataRow

        assert self.type in ["pandas", "numpy", "array"], f"Invalid type: {type}"

    @override
    def build(self):
        self.columns = [ft.DataColumn(header) for header in self.headers]
        self.rows = [
            ft.DataRow([ft.DataCell(value=cell, on_tap=self.on_select) for cell in row])
            for row in self.value
        ]
        self.element = ft.Column(
            [
                ft.Text(self.label, visible=self.show_label),
                ft.DataTable(
                    columns=self.columns,
                    rows=self.rows,
                    disabled=not self.interactive,
                    overflow_row_behaviour=self.overflow_row_behaviour,
                    wrap=self.wrap,
                    on_select_all=self.on_select,
                ),
            ],
            key=self.elem_id,
            visible=self.visible,
        )
        self.value_thread.start()
        return self.element


class ReadOnlyText(Component):
    def __init__(
        self,
        value,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.value = value
        self.label = label
        self.every = every if every else 1
        self.show_label = show_label
        self.interactive = interactive or True
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.element: ft.Column

    @override
    def build(self):
        self.text = ft.Text(
            self.value,
            key=self.elem_id,
            visible=self.visible,
            disabled=not self.interactive,
        )
        self.element = ft.Column(
            [ft.Text(self.label, visible=self.show_label), self.text]
        )
        self.value_thread.start()
        return self.element

    @override
    def get_value(self):
        return self.text.value

    @override
    def set_value(self, value: str | Callable | None):
        if callable(value):
            self.value = value
            return
        assert isinstance(value, (str, NoneType)), f"Invalid value: {value}"
        self.text.value = value
        self.text.update()


class Code(TextBox):
    pass


class Markdown(Component, Changeable):
    """
    Used to render arbitrary Markdown output. Can also render latex enclosed by dollar signs.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a valid {str} that can be rendered as Markdown.

    Demos: blocks_hello, blocks_kinematics
    Guides: key-features
    """

    def __init__(
        self,
        value: str | Callable = "",
        *,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Value to show in Markdown component. If callable, the function will be called whenever the app loads to set the initial value of the component.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs)
        self.value = value
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = elem_classes
        self.element: ft.Markdown

    @override
    def build(self):
        self.element = ft.Markdown(
            key=self.elem_id,
            visible=self.visible,
            value=self.value,
        )
        self.value_thread.start()
        return self.element


class HTML(Markdown):
    pass


class TextArea(TextBox):
    pass


class Video(
    Component,
    Changeable,
    Clearable,
    Playable,
    Uploadable,
):
    pass


class Audio(
    Component,
    Changeable,
    Clearable,
    Playable,
    Streamable,
    Uploadable,
):
    pass


class HighlightedText(ReadOnlyText):
    pass


class Json(Markdown):
    @override
    def set_value(self, json_value: str | dict | list | Callable | None):
        if callable(json_value):
            self.value = json_value
            return
        if isinstance(json_value, str):
            self.element.value = json_value
        else:
            self.element.value = json.dumps(json_value, indent=2)
        self.element.value = f"```json\n{self.element.value}\n```"
        self.element.update()


class State:
    def __init__(self, value, **kwargs) -> None:
        self.value = deepcopy(value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class Variable(State):
    pass


Text = Textbox = TextBox
Checkbox = CheckBox
Checkboxgroup = CheckboxGroup
Dataframe = DataFrame
Dropdown = DropDown
JSON = Json

Highlightedtext = HighlightedText
Highlight = HighlightedText
