from typing import Optional
from typing_extensions import override
from gradio.components import Component, Container, Selectable
import flet as ft


class Row(Container):
    """
    Row is a layout element within Blocks that renders all children horizontally.
    Example:
        with gr.Blocks() as demo:
            with gr.Row():
                gr.Image("lion.jpg")
                gr.Image("tiger.jpg")
        demo.launch()
    Guides: controlling-layout
    """

    def __init__(
        self,
        *,
        variant: str = "default",
        visible: bool = True,
        elem_id: str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            variant: row type, 'default' (no background), 'panel' (gray background color and rounded corners), or 'compact' (rounded corners and no internal gap).
            visible: If False, row will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs)
        self.variant = variant
        self.visible = visible
        self.elem_id = elem_id
        self.element: ft.Container
        assert self.variant in [
            "default",
            "panel",
            "compact",
        ], f"Invalid variant {self.variant}"
        self.bgcolor = "#f5f5f5" if variant == "panel" else None
        self.shape = ft.BoxShape.CIRCLE if variant != "default" else None
        self.border = ft.border.all(1) if variant != "default" else None
        self.border_radius = 5 if variant != "default" else None

    @override
    def build(self):
        self.element = ft.Container(
            ft.Row(
                self.build_children(),
                spacing=0 if self.variant == "compact" else 16,
            ),
            visible=self.visible,
            key=self.elem_id,
            bgcolor=self.bgcolor,
            shape=self.shape,
            border=self.border,
            border_radius=self.border_radius,
        )
        return self.element

    @override
    def value_callback(self):
        pass


class Column(Row):
    """
    Column is a layout element within Blocks that renders all children vertically. The widths of columns can be set through the `scale` and `min_width` parameters.
    If a certain scale results in a column narrower than min_width, the min_width parameter will win.
    Example:
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=1):
                    text1 = gr.Textbox()
                    text2 = gr.Textbox()
                with gr.Column(scale=4):
                    btn1 = gr.Button("Button 1")
                    btn2 = gr.Button("Button 2")
    Guides: controlling-layout
    """

    def __init__(
        self,
        *,
        scale: int = 1,
        min_width: int = 320,
        variant: str = "default",
        visible: bool = True,
        elem_id: str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            scale: relative width compared to adjacent Columns. For example, if Column A has scale=2, and Column B has scale=1, A will be twice as wide as B.
            min_width: minimum pixel width of Column, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in a column narrower than min_width, the min_width parameter will be respected first.
            variant: column type, 'default' (no background), 'panel' (gray background color and rounded corners), or 'compact' (rounded corners and no internal gap).
            visible: If False, column will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs, variant=variant, visible=visible, elem_id=elem_id)
        self.scale = scale
        self.min_width = min_width

    @override
    def build(self):
        self.element = ft.Container(
            ft.Column(
                self.build_children(),
                spacing=0 if self.variant == "compact" else 16,
                expand=True,
            ),
            # width=self.min_width * self.scale,
            visible=self.visible,
            key=self.elem_id,
            bgcolor=self.bgcolor,
            shape=self.shape,
            border=self.border,
            border_radius=self.border_radius,
            expand=self.kwargs.get("expand", False),
        )
        return self.element


class Tabs(Container, Selectable):
    """
    Tabs is a layout element within Blocks that can contain multiple "Tab" Components.
    """

    current: "Tabs"

    def __init__(
        self,
        *,
        selected: int | str | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            selected: The currently selected tab. Must correspond to an id passed to the one of the child TabItems. Defaults to the first TabItem.
            visible: If False, Tabs will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs)
        Tabs.current = self
        self.selected = selected or 0
        self.visible = visible
        self.elem_id = elem_id
        self.element: ft.Tabs
        self.local_scope: list[Tab]

    def locate_index(self, tab_name: str):
        return next(
            i for i, tab in enumerate(self.local_scope) if tab.get_value() == tab_name
        )

    @override
    def build(self):
        if isinstance(self.selected, str):
            self.selected = self.locate_index(self.selected)
        assert all(map(lambda x: isinstance(x, Tab), self.children)), (
            "All children of Tabs must be Tab elements. "
            f"Got {self.children} instead."
        )

        self.element = ft.Tabs(
            tabs=self.build_children(),
            visible=self.visible,
            key=self.elem_id,
            selected_index=self.selected,
            on_change=self.on_select,
            expand=True,
        )
        return self.element

    @override
    def get_value(self):
        return self.element.selected_index

    @override
    def set_value(self, value):
        self.element.selected_index = value
        self.element.update()


class Tab(Container, Selectable):
    """
    Tab (or its alias TabItem) is a layout element. Components defined within the Tab will be visible when this tab is selected tab.
    Example:
        with gr.Blocks() as demo:
            with gr.Tab("Lion"):
                gr.Image("lion.jpg")
                gr.Button("New Lion")
            with gr.Tab("Tiger"):
                gr.Image("tiger.jpg")
                gr.Button("New Tiger")
    Guides: controlling-layout
    """

    def __init__(
        self,
        label: str,
        *,
        id: int | str | None = None,
        elem_id: str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            label: The visual label for the tab
            id: An optional identifier for the tab, required if you wish to control the selected tab from a predict function.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs)
        self.label = label
        self.id = id
        self.elem_id = elem_id
        self.element: ft.Tab

    # @override
    # def append_to_scope(self):
    #     scope = Container.current_scope
    #     self.parent = Container.current_scope = Tabs.current
    #     super().append_to_scope()
    #     Container.current_scope = scope

    @override
    def build(self):
        assert isinstance(
            self.parent, Tabs
        ), f"Tab must be a child of Tabs, not {self.parent}"
        self.element = ft.Tab(
            text=self.label,
            content=ft.ListView(
                self.build_children(),
                expand=True,
                key=self.elem_id,
            ),
        )
        return self.element

    @override
    def get_value(self):
        return self.element.text

    @override
    def set_value(self, value):
        self.element.text = value
        self.element.update()


class Accordion(Container):
    """
    Accordion is a layout element which can be toggled to show/hide the contained content.
    Example:
        with gr.Accordion("See Details"):
            gr.Markdown("lorem ipsum")
    """

    def __init__(
        self,
        label,
        *,
        open: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            label: name of accordion section.
            open: if True, accordion is open by default.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs)
        self.label = label
        self.open = open
        self.visible = visible
        self.elem_id = elem_id

    @override
    def build(self):
        self.sub = ft.Column(
            self.build_children(),
            visible=self.open,
        )
        self.toggle_btn = ft.TextButton(
            icon=ft.icons.ARROW_DROP_DOWN if self.open else ft.icons.ARROW_RIGHT,
            text=self.label,
            on_click=self.toggle,
        )
        self.element = ft.Column(
            [
                self.toggle_btn,
                self.sub,
            ],
            visible=self.visible,
            key=self.elem_id,
            spacing=8,
        )
        return self.element

    @override
    def toggle(self, *_, open=None):
        self.open = not self.open if open is None else open
        self.sub.visible = self.open
        self.toggle_btn.icon = (
            ft.icons.ARROW_DROP_DOWN if self.open else ft.icons.ARROW_RIGHT
        )
        self.element.update()

    @override
    def get_value(self):
        return self.open

    @override
    def set_value(self, value):
        self.toggle(value)


class Group(Container):
    """
    Group is a layout element within Blocks which groups together children so that
    they do not have any padding or margin between them.
    Example:
        with gr.Group():
            gr.Textbox(label="First")
            gr.Textbox(label="Last")
    """

    def __init__(
        self,
        *,
        visible: bool = True,
        elem_id: str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            visible: If False, group will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        super().__init__(**kwargs)
        self.visible = visible
        self.elem_id = elem_id

    @override
    def build(self):
        self.element = ft.ListView(
            self.build_children(),
            visible=self.visible,
            key=self.elem_id,
            spacing=0,
        )
        return self.element


class Box(Column):
    pass


TabItem = Tab
