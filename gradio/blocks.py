import asyncio
import json
import os
import threading
import traceback
from typing import Any, Callable, Optional
import anyio

import flet as ft
import yaml
from fastapi import FastAPI
from typing_extensions import override

from gradio.components import CallbackGroup, CallbackIO, Component, Container
from gradio.layouts import Column, Row
from gradio.themes import Theme
from gradio.utils import logger
from gradio.warnings import warn_kwargs


class DummyApp(FastAPI):
    pass


class DummyServer:
    def __setattr__(self, __name: str, __value: Any) -> None:
        pass


class DummyLocalScope:
    def append(self, _):
        pass


class Blocks(Column):
    """
    Blocks is Gradio's low-level API that allows you to create more custom web
    applications and demos than Interfaces (yet still entirely in Python).


    Compared to the Interface class, Blocks offers more flexibility and control over:
    (1) the layout of components (2) the events that
    trigger the execution of functions (3) data flows (e.g. inputs can trigger outputs,
    which can trigger the next level of outputs). Blocks also offers ways to group
    together related demos such as with tabs.


    The basic usage of Blocks is as follows: create a Blocks object, then use it as a
    context (with the "with" statement), and then define layouts, components, or events
    within the Blocks context. Finally, call the launch() method to launch the demo.

    Example:
        import gradio as gr
        def update(name):
            return f"Welcome to Gradio, {name}!"

        with gr.Blocks() as demo:
            gr.Markdown("Start typing below and then click **Run** to see the output.")
            with gr.Row():
                inp = gr.Textbox(placeholder="What is your name?")
                out = gr.Textbox()
            btn = gr.Button("Run")
            btn.click(fn=update, inputs=inp, outputs=out)

        demo.launch()
    Demos: blocks_hello, blocks_flipper, blocks_speech_text_sentiment, generate_english_german, sound_alert
    Guides: blocks-and-event-listeners, controlling-layout, state-in-blocks, custom-CSS-and-JS, custom-interpretations-with-blocks, using-blocks-like-functions
    """

    def __init__(
        self,
        theme: Theme | str | None = None,
        analytics_enabled: bool | None = None,
        mode: str = "blocks",
        title: str = "Gradio",
        css: str | None = None,
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
        super().__init__(**kwargs)
        self.load_callback = CallbackGroup()
        self.theme = theme
        self.analytics_enabled = analytics_enabled
        self.mode = mode
        self.title = title
        self.css = css

    def render(self):
        pass

    def load(self, fn, *args, **kwargs):
        self.load_callback.append(fn)

    def launch(
        self,
        inline: bool | None = None,
        inbrowser: bool = False,
        share: bool | None = None,
        debug: bool = False,
        enable_queue: bool | None = None,
        max_threads: int = 40,
        auth: Callable | tuple[str, str] | list[tuple[str, str]] | None = None,
        auth_message: str | None = None,
        prevent_thread_lock: bool = False,
        show_error: bool = False,
        server_name: str | None = None,
        server_port: int | None = None,
        show_tips: bool = False,
        height: int = 500,
        width: int | str = "100%",
        encrypt: bool | None = None,
        favicon_path: str | None = None,
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
        ssl_keyfile_password: str | None = None,
        ssl_verify: bool = True,
        quiet: bool = False,
        show_api: bool = True,
        file_directories: list[str] | None = None,
        allowed_paths: list[str] | None = None,
        blocked_paths: list[str] | None = None,
        root_path: str = "",
        _frontend: bool = True,
        app_kwargs: dict[str, Any] | None = None,
    ) -> tuple[FastAPI, str, str]:
        """
        Launches a simple web server that serves the demo. Can also be used to create a
        public link used by anyone to access the demo from their browser by setting share=True.

        Parameters:
            inline: whether to display in the interface inline in an iframe. Defaults to True in python notebooks; False otherwise.
            inbrowser: whether to automatically launch the interface in a new tab on the default browser.
            share: whether to create a publicly shareable link for the interface. Creates an SSH tunnel to make your UI accessible from anywhere. If not provided, it is set to False by default every time, except when running in Google Colab. When localhost is not accessible (e.g. Google Colab), setting share=False is not supported.
            debug: if True, blocks the main thread from running. If running in Google Colab, this is needed to print the errors in the cell output.
            auth: If provided, username and password (or list of username-password tuples) required to access interface. Can also provide function that takes username and password and returns True if valid login.
            auth_message: If provided, HTML message provided on login page.
            prevent_thread_lock: If True, the interface will block the main thread while the server is running.
            show_error: If True, any errors in the interface will be displayed in an alert modal and printed in the browser console log
            server_port: will start gradio app on this port (if available). Can be set by environment variable GRADIO_SERVER_PORT. If None, will search for an available port starting at 7860.
            server_name: to make app accessible on local network, set this to "0.0.0.0". Can be set by environment variable GRADIO_SERVER_NAME. If None, will use "127.0.0.1".
            show_tips: if True, will occasionally show tips about new Gradio features
            enable_queue: DEPRECATED (use .queue() method instead.) if True, inference requests will be served through a queue instead of with parallel threads. Required for longer inference times (> 1min) to prevent timeout. The default option in HuggingFace Spaces is True. The default option elsewhere is False.
            max_threads: the maximum number of total threads that the Gradio app can generate in parallel. The default is inherited from the starlette library (currently 40). Applies whether the queue is enabled or not. But if queuing is enabled, this parameter is increaseed to be at least the concurrency_count of the queue.
            width: The width in pixels of the iframe element containing the interface (used if inline=True)
            height: The height in pixels of the iframe element containing the interface (used if inline=True)
            encrypt: DEPRECATED. Has no effect.
            favicon_path: If a path to a file (.png, .gif, or .ico) is provided, it will be used as the favicon for the web page.
            ssl_keyfile: If a path to a file is provided, will use this as the private key file to create a local server running on https.
            ssl_certfile: If a path to a file is provided, will use this as the signed certificate for https. Needs to be provided if ssl_keyfile is provided.
            ssl_keyfile_password: If a password is provided, will use this with the ssl certificate for https.
            ssl_verify: If False, skips certificate validation which allows self-signed certificates to be used.
            quiet: If True, suppresses most print statements.
            show_api: If True, shows the api docs in the footer of the app. Default True. If the queue is enabled, then api_open parameter of .queue() will determine if the api docs are shown, independent of the value of show_api.
            file_directories: This parameter has been renamed to `allowed_paths`. It will be removed in a future version.
            allowed_paths: List of complete filepaths or parent directories that gradio is allowed to serve (in addition to the directory containing the gradio python file). Must be absolute paths. Warning: if you provide directories, any files in these directories or their subdirectories are accessible to all users of your app.
            blocked_paths: List of complete filepaths or parent directories that gradio is not allowed to serve (i.e. users of your app are not allowed to access). Must be absolute paths. Warning: takes precedence over `allowed_paths` and all other directories exposed by Gradio by default.
            root_path: The root path (or "mount point") of the application, if it's not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application. For example, if the application is served at "https://example.com/myapp", the `root_path` should be set to "/myapp".
            app_kwargs: Additional keyword arguments to pass to the underlying FastAPI app as a dictionary of parameter keys and argument values. For example, `{"docs_url": "/docs"}`
        Returns:
            app: FastAPI app object that is running the demo
            local_url: Locally accessible link to the demo
            share_url: Publicly accessible link to the demo (if share=True, otherwise None)
        Example: (Blocks)
            import gradio as gr
            def reverse(text):
                return text[::-1]
            with gr.Blocks() as demo:
                button = gr.Button(value="Reverse")
                button.click(reverse, gr.Textbox(), gr.Textbox())
            demo.launch(share=True, auth=("username", "password"))
        Example:  (Interface)
            import gradio as gr
            def reverse(text):
                return text[::-1]
            demo = gr.Interface(reverse, "text", "text")
            demo.launch(share=True, auth=("username", "password"))
        """

        def main(page: ft.Page):
            try:
                page.title = self.title
                page.window_min_width = (
                    int(10.0 * int(width[:-1])) if isinstance(width, str) else width
                )
                page.window_min_height = height
                page.expand = True
                page.padding = ft.padding.all(16)
                page.theme = ft.Theme(
                    use_material3=True,
                    visual_density=ft.ThemeVisualDensity.ADAPTIVEPLATFORMDENSITY,
                )
                page.add(Container.current_scope.build())  # type: ignore
                with open("debug_layout.yml", "w") as f:
                    f.write(
                        yaml.dump(
                            Container.current_scope.debug_info(),  # type: ignore
                            indent=2,
                            sort_keys=False,
                        )
                    )
            except Exception as e:
                raise e

        ft.app(target=main, name=self.title)
        os._exit(0)
        return DummyApp(), "GUI", "GUI"

    def queue(self, *_, **__):
        pass

    @property
    def fns(self):
        return []

    @property
    def server(self):
        return DummyServer()

    # def __getattr__(self, __name: str) -> Any:
    #     if __name in self.__dict__:
    #         return self.__dict__.get(__name)
    #     else:
    #         logger.error(f'"{__name}" is not a valid attribute of this Blocks object.')
    #         return None


class Interface(Component):
    def __init__(
        self,
        fn: Callable,
        inputs: CallbackIO | str | list[str],
        outputs: CallbackIO | str | list[str],
        theme: Theme | str | None = None,
        description: Optional[str] = None,
        title: str = "Gradio",
        **kwargs,
    ):
        Container.current_scope = self  # type: ignore
        self.local_scope = DummyLocalScope()
        super().__init__(theme, **kwargs)
        if not isinstance(inputs, list):
            inputs = [inputs]  # type: ignore
        if not isinstance(outputs, list):
            outputs = [outputs]  # type: ignore
        self.inputs = [Component.dynamic(i) for i in inputs]  # type: ignore
        self.outputs = [Component.dynamic(o) for o in outputs]  # type: ignore
        self.value = None
        self.title = title
        self.description = description
        self.element: Row
        self.fn = fn
        for i in self.inputs:
            i.parent = self  # for debug
        for o in self.outputs:
            o.parent = self  # for debug
            o.interactive = False  # disable input for outputs

    @override
    def build(self):
        inputs = [e.build() for e in self.inputs]
        outputs = [e.build() for e in self.outputs]
        inputs.append(
            ft.Row(
                [
                    ft.ElevatedButton(
                        "Clear",
                        on_click=self.clear,
                    ),
                    ft.ElevatedButton(
                        "Submit",
                        on_click=lambda _: self.set_value(self.fn(*self.get_value())),
                    ),
                ],
                expand=True,
            )
        )
        self.element = ft.Column(
            [
                ft.Text(
                    self.description,
                    visible=self.description is not None,
                ),
                ft.Row(
                    [
                        ft.Column(inputs, expand=True, spacing=8),
                        ft.Column(outputs, expand=True, spacing=8),
                    ],
                    expand=True,
                ),
            ],
            expand=True,
            spacing=16,
        )
        # self.value_thread.start()
        return self.element

    def clear(self, *_):
        for i in self.inputs:
            i.set_value(None)
        for o in self.outputs:
            o.set_value(None)

    @override
    def get_value(self):
        return [e.get_value() for e in self.inputs]

    @override
    def set_value(self, value):
        for e, v in zip(self.outputs, value):
            e.set_value(v)

    @override
    def debug_info(self):
        return {
            "type": "Interface",
            "inputs": [i.debug_info() for i in self.inputs],
            "outputs": [o.debug_info() for o in self.outputs],
        }

    launch = Blocks.launch


class Block(Component):
    pass


class BlockContext(Component):
    pass
