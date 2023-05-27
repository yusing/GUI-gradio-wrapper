from abc import abstractmethod
from collections import abc
from copy import deepcopy
from functools import wraps
from inspect import Traceback, getframeinfo, stack
import os
import random
import traceback
from typing_extensions import override
import warnings
from typing import Any, Callable, ContextManager, Iterable, Optional

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image as _Image
import requests

from gradio.utils import hex_to_rgba, logger
from gradio.warnings import warn_kwargs, warn_unimplemented


class CallbackGroup:
    def __init__(self) -> None:
        self.callbacks: list[Callable] = []

    def __call__(self) -> list[Any]:
        return [fn() for fn in self.callbacks]

    def append(self, fn: Optional[Callable]):
        if fn is None:
            return
        self.callbacks.append(fn)


class Component:
    current_scope: Optional["Container"] = None
    previous_scopes: list["Container"] = []
    component_map: dict[str | int, "Component"] = {}

    def __init__(
        self, *args, add_to_scope=True, excluded_attr: list[str] = [], **kwargs
    ) -> None:
        self.tag = kwargs.pop("elem_id", None)
        if self.tag is None:
            self.tag = dpg.generate_uuid()
        else:
            self.tag = f"{self.__class__.__name__}_{self.tag}"
        self.default_value = kwargs.pop("value", args[0] if any(args) else None)
        if not kwargs.pop("visible", True):
            self.show = False
        if not kwargs.pop("interactive", True):
            self.enabled = False
        self.label = kwargs.get("label")
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        self.user_data = kwargs
        self.user_data["local_scope"] = []
        self.user_data["traceback"] = stack(1)[2]
        self.callback = CallbackGroup()
        self.drag_callback = CallbackGroup()
        for attr in excluded_attr:
            self.__dict__.pop(attr, None)
        if isinstance(self, Window):
            Component.current_scope = self
        elif add_to_scope:
            self.append_to_scope()

    @abstractmethod
    def build_element(self) -> int | str:
        assert not isinstance(
            self, Container
        ), "Call build_container instead for Container"
        raise NotImplementedError

    def append_to_scope(self):
        if Component.current_scope is None:
            Component.current_scope = Window()
        Component.current_scope.local_scope.append(self)
        if self.tag in Component.component_map:
            self.tag = f"{self.tag}_{dpg.generate_uuid()}"
        Component.component_map[self.tag] = self

    def click(self, *args, **kwargs):
        self.callback.append(kwargs.get("fn", args[0] if any(args) else None))
        return self

    def release(self, *args, **kwargs):
        self.drag_callback.append(kwargs.get("fn", args[0] if any(args) else None))
        return self

    @warn_unimplemented
    def change(self, *args, **kwargs):
        return self

    @warn_unimplemented
    def style(self, *args, **kwargs):
        return self

    # @warn_unimplemented
    # def input(self, *args, **kwargs):
    #     return self

    # @warn_unimplemented
    # def edit(self, *args, **kwargs):
    #     return self

    # @warn_unimplemented
    # def clear(self, *args, **kwargs):
    #     return self
    input = edit = clear = then = click

    @warn_unimplemented
    def submit(self, *args, **kwargs):
        return self

    @warn_unimplemented
    def select(self, *args, **kwargs):
        return self

    @property
    def value(self):
        if dpg.does_item_exist(self.tag):
            return dpg.get_value(self.tag)
        return getattr(self, "default_value", None)

    @value.setter
    def value(self, value):
        if dpg.does_item_exist(self.tag):
            dpg.set_value(self.tag, value)
        else:
            self.default_value = value

    @property
    def visible(self):
        if dpg.does_item_exist(self.tag):
            return dpg.is_item_visible(self.tag)
        return getattr(self, "show", True)

    @visible.setter
    def visible(self, value):
        if dpg.does_item_exist(self.tag):
            dpg.configure_item(self.tag, show=value)
        else:
            self.show = value

    @property
    def local_scope(self) -> list["Component"]:
        return self.user_data["local_scope"]

    @property
    def elem_id(self):
        return self.tag

    @property
    def traceback(self) -> Traceback:
        return self.user_data["traceback"]

    def __enter__(self):
        assert isinstance(
            self, Container
        ), "Only Container can be used as context manager"
        Component.previous_scopes.append(Component.current_scope)
        Component.current_scope = self
        return self

    def __exit__(self, *_):
        Component.current_scope = Component.previous_scopes.pop()
        assert Component.current_scope is not None, "No previous scope to exit to"

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"<{self.__class__.__name__} tag={self.tag} caller={os.path.basename(self.traceback.filename)}:{self.traceback.lineno}>"

    def debug_info(self):
        info = {
            "element": repr(self),
        }
        if any(self.local_scope):
            info["children"] = [ele.debug_info() for ele in self.local_scope]
        if "build_error" in self.user_data:
            info["build_error"] = self.user_data.get("build_error")
        return info

    @staticmethod
    def build(scope: Optional["Container"] = None):
        from gradio.layouts import Tabs, Tab

        if scope is None:
            assert Component.current_scope, "No scope to build"
            scope = Component.current_scope
        assert isinstance(scope, Container), f"Invalid container {scope}"
        try:
            with scope.build_container():
                for ele in scope.local_scope:
                    if isinstance(ele, Container):
                        if isinstance(ele, Tab) and not isinstance(scope, Tabs):
                            with dpg.tab_bar(show=False):
                                Component.build(ele)
                        elif isinstance(scope, Tabs) and not isinstance(ele, Tab):
                            with dpg.tab(show=False):
                                Component.build(ele)
                        else:
                            Component.build(ele)
                    else:
                        ele.__dict__.pop("custom_script_source", None)
                        try:
                            ele.build_element()
                        except Exception as e:
                            ele.user_data["build_error"] = {
                                "error": str(e),
                                "element": ele.__class__.__name__,
                                "traceback": traceback.format_exc().split("\n"),
                            }
                            return
                        assert not any(ele.local_scope), "Non-root element has children"
        except Exception as e:
            scope.user_data["build_error"] = {
                "error": str(e),
                "element": scope.__class__.__name__,
                "traceback": traceback.format_exc().split("\n"),
            }
            raise e


class Container(Component):
    def __init__(self, *args, excluded_attr: list[str] = [], **kwargs) -> None:
        excluded_attr.extend(["enabled", "default_value"])
        super().__init__(*args, **kwargs, excluded_attr=excluded_attr)

    def build_container(self) -> ContextManager[int | str]:
        raise NotImplementedError


class Window(Container):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def build_container(self) -> ContextManager[int | str]:
        return dpg.window(no_title_bar=True, autosize=True)


class IOComponent(Component):
    pass


class Text(Component):
    """
    Creates a textarea for user to enter string input or display string output.
    Preprocessing: passes textarea value as a {str} into the function.
    Postprocessing: expects a {str} returned from function and sets textarea value to it.
    Examples-format: a {str} representing the textbox input.

    Demos: hello_world, diff_texts, sentence_builder
    Guides: creating-a-chatbot, real-time-speech-recognition
    """

    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs, excluded_attr=["enabled"])

    @override
    def build_element(self):
        return dpg.add_input_text(**self.__dict__)


class Number(Component):
    """
    Creates a numeric field for user to enter numbers as input or display numeric output.
    Preprocessing: passes field value as a {float} or {int} into the function, depending on `precision`.
    Postprocessing: expects an {int} or {float} returned from the function and sets field value to it.
    Examples-format: a {float} or {int} representing the number's value.

    Demos: tax_calculator, titanic_survival, blocks_simple_squares
    """

    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs)

    @override
    def build_element(self):
        readonly = self.__dict__.pop("enabled", False)
        if readonly:
            return dpg.add_input_float(**self.__dict__)
        else:
            self.__dict__.pop("callback", None)
            return dpg.add_text(**self.__dict__)


class Image(Component):
    """
    Creates an image component that can be used to upload/draw images (as an input) or display images (as an output).
    Preprocessing: passes the uploaded image as a {numpy.array}, {PIL.Image} or {str} filepath depending on `type` -- unless `tool` is `sketch` AND source is one of `upload` or `webcam`. In these cases, a {dict} with keys `image` and `mask` is passed, and the format of the corresponding values depends on `type`.
    Postprocessing: expects a {numpy.array}, {PIL.Image} or {str} or {pathlib.Path} filepath to an image and displays the image.
    Examples-format: a {str} filepath to a local file that contains the image.
    Demos: image_mod, image_mod_default_image
    Guides: image-classification-in-pytorch, image-classification-in-tensorflow, image-classification-with-vision-transformers, building-a-pictionary_app, create-your-own-friends-with-a-gan
    """

    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        # TODO: Handle crop and resize to `shape`
        # TODO: Handle invert_colors
        # TODO: Handle source
        if isinstance(self.default_value, np.ndarray):
            self.default_value = _Image.fromarray(self.default_value)
        elif isinstance(self.default_value, str):
            if os.path.exists(self.default_value):
                self.default_value = _Image.open(self.default_value)
            elif self.default_value.startswith("http"):
                self.default_value = _Image.open(
                    requests.get(self.default_value, stream=True).content
                )
            else:
                raise ValueError(f"Invalid image path: {self.default_value}")
        elif isinstance(self.default_value, _Image.Image) or self.default_value is None:
            pass
        else:
            raise ValueError(f"Invalid image type: {type(self.default_value)}")
        self.default_value = (
            self.default_value.convert("RGB")
            if self.default_value
            else _Image.new("RGB", getattr(self, "shape", (300, 300)))
        )
        with dpg.texture_registry():
            self.texture_tag = dpg.add_static_texture(
                width=self.default_value.width,
                height=self.default_value.height,
                default_value=self.default_value,
                tag=f"{self.tag}_texture",
            )
        del self.default_value

    @override
    def build_element(self) -> int | str:
        self.__dict__.pop("callback", None)
        return dpg.add_image(**self.__dict__)


class CheckboxGroup(Component):
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
        *args,
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
        self.choices = choices if choices else []
        super().__init__(*args, **kwargs)

    @override
    def build_element(self):
        with dpg.group(tag=self.tag, label=self.label, horizontal=True):
            for choice in self.choices:
                dpg.add_checkbox(
                    label=choice,
                    tag=f"{self.tag}_{choice}",
                    default_value=choice
                    in self.default_value,  # TODO: check if value is list[Callable]
                )
        return self.tag


class Radio(Component):
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
        *args,
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
        self.items = choices if choices else []
        super().__init__(*args, **kwargs)

    @override
    def build_element(self):
        return dpg.add_radio_button(
            horizontal=True,
            **self.__dict__,
        )


class ColorPicker(Component):
    """
    Creates a color picker for user to select a color as string input.
    Preprocessing: passes selected color value as a {str} into the function.
    Postprocessing: expects a {str} returned from function and sets color picker value to it.
    Examples-format: a {str} with a hexadecimal representation of a color, e.g. "#ff0000" for red.
    Demos: color_picker, color_generator
    """

    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs)

    @override
    def build_element(self):
        if isinstance(self.default_value, str):
            self.default_value = hex_to_rgba(self.default_value)
        elif callable(self.default_value):
            self.default_value = hex_to_rgba(self.default_value())
        return dpg.add_color_picker(
            **self.__dict__, no_label=not self.user_data.get("show_label", True)
        )


class DropDown(Component):
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
        *args,
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
        super().__init__(*args, **kwargs)
        self.items = choices if choices else []

    @override
    def build_element(self):
        self.__dict__.pop("init_field", None)
        return dpg.add_combo(**self.__dict__)


class Label(Component):
    """
    Displays a classification label, along with confidence scores of top categories, if provided.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a {Dict[str, float]} of classes and confidences, or {str} with just the class or an {int}/{float} for regression outputs, or a {str} path to a .json file containing a json dictionary in the structure produced by Label.postprocess().

    Demos: main_note, titanic_survival
    Guides: image-classification-in-pytorch, image-classification-in-tensorflow, image-classification-with-vision-transformers, building-a-pictionary-app
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, excluded_attr=["callback"])

    @override
    def build_element(self):
        return dpg.add_text(**self.__dict__)


class File(Component):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, excluded_attr=["drag_callback", "enabled"])
        self.default_path = self.__dict__.pop("default_value", "")

    @override
    def build_element(self):
        if callable(self.default_path):
            self.default_path = self.default_path()
        return dpg.add_file_dialog(**self.__dict__)


class Slider(Component):
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
        self.no_input = not kwargs.pop("interactive", True)
        super().__init__(**kwargs)
        self.min_value = minimum
        self.max_value = maximum
        self.default_value = value

    @override
    def build_element(self):
        if callable(self.default_value):
            self.default_value = self.default_value()
        if self.default_value is None or self.user_data.get(
            "randomize", self.default_value is None
        ):
            self.default_value = random.uniform(self.min_value, self.max_value)
        return dpg.add_slider_float(**self.__dict__, clamped=True)


class Button(Component):
    """
    Used to create a button, that can be assigned arbitrary click() events. The label (value) of the button can be used as an input or set via the output of a function.

    Preprocessing: passes the button value as a {str} into the function
    Postprocessing: expects a {str} to be returned from a function, which is set as the label of the button
    Demos: blocks_inputs, blocks_kinematics
    """

    def __init__(
        self,
        *args,
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
        super().__init__(*args, **kwargs, excluded_attr=["default_value"])

    @override
    def build_element(self):
        return dpg.add_button(**self.__dict__)


class CheckBox(Component):
    @override
    def build_element(self):
        if callable(self.default_value):
            self.default_value = self.default_value()
        elif self.default_value is None:
            self.default_value = False
        assert isinstance(self.default_value, bool), "Invalid default value"
        return dpg.add_checkbox(**self.__dict__)


class Gallery(Component):
    @override
    def build_element(self) -> int | str:
        self.__dict__.pop("callback", None)
        images = self.__dict__.pop("default_value")
        if callable(images):
            images = images()
        elif images is None:
            images = []
        images = list(map(Image, images))
        with dpg.group(**self.__dict__):
            for image in images:
                image.build_element()
        return self.tag


class DataFrame(Component):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self, "headers"):
            self.headers: list[str] = []

    @override
    def build_element(self) -> int | str:
        """
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
        """
        with dpg.table(
            header_row=True,
            tag=self.tag,
            show=self.show,
            enabled=self.enabled,
            **self.__dict__,
        ):
            for header in self.headers:
                dpg.add_table_column(label=header)
            for row in self.default_value:
                with dpg.table_row():
                    for col in row:
                        dpg.add_text(str(col))
        return self.tag


class State:
    def __init__(self, value, **kwargs) -> None:
        self.value = deepcopy(value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class Variable(State):
    pass


class Code(Text):
    pass


class HTML(Text):
    pass


class TextArea(Text):
    pass


class Video(Text):
    pass


class Audio:
    @warn_unimplemented
    def __init__(self, *args, **kwargs) -> None:
        pass


class HighlightedText(Text):
    pass


class Json(Text):
    pass


Checkbox = CheckBox
Checkboxgroup = CheckboxGroup
Dataframe = DataFrame
Dropdown = DropDown
Textbox = Text
Markdown = HTML
JSON = Json

Highlightedtext = HighlightedText
Highlight = HighlightedText
