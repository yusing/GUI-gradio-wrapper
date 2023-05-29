import faulthandler

faulthandler.enable()
import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.configure_app(manual_callback_management=True)
with dpg.font_registry():
    import os

    if os.name == "nt":
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)

    roboto_medium = dpg.add_font(
        os.path.join(os.path.dirname(__file__), "fonts/RobotoMono-Medium.ttf"),
        16 * 2,
    )
    dpg.bind_font(roboto_medium)
    dpg.set_global_font_scale(0.5)

import gradio.processing_utils
from gradio.blocks import Block, Blocks
from gradio.components import (  # JSON,; AnnotatedImage,; Annotatedimage,; BarPlot,; Carousel,; Chatbot,; Dataset,; Interpretation,; LinePlot,; Model3D,; Plot,; ScatterPlot,; StatusTracker,; TimeSeries,; Timeseries,; UploadButton,
    HTML,
    JSON,
    Audio,
    Button,
    Checkbox,
    CheckBox,
    CheckboxGroup,
    Checkboxgroup,
    Code,
    ColorPicker,
    DataFrame,
    Dataframe,
    DropDown,
    Dropdown,
    File,
    Gallery,
    Highlight,
    HighlightedText,
    Highlightedtext,
    Image,
    Json,
    Label,
    Markdown,
    Number,
    Radio,
    Slider,
    State,
    Text,
    Textbox,
    Variable,
    Video,
)
from gradio.helpers import update
from gradio.layouts import Accordion, Box, Column, Group, Row, Tab, TabItem, Tabs
from gradio.routes import mount_gradio_app
from gradio.templates import (
    Files,
    ImageMask,
    ImagePaint,
    List,
    Matrix,
    Mic,
    Microphone,
    Numpy,
    Paint,
    Pil,
    PlayableVideo,
    Sketchpad,
    TextArea,
    Webcam,
)
from gradio.themes import Theme

__version__ = "3.32.0"
