def update(**kwargs) -> dict:
    """
    Updates component properties. When a function passed into a Gradio Interface or a Blocks events returns a typical value, it updates the value of the output component. But it is also possible to update the properties of an output component (such as the number of lines of a `Textbox` or the visibility of an `Image`) by returning the component's `update()` function, which takes as parameters any of the constructor parameters for that component.
    This is a shorthand for using the update method on a component.
    For example, rather than using gr.Number.update(...) you can just use gr.update(...).
    Note that your editor's autocompletion will suggest proper parameters
    if you use the update method on the component.
    Demos: blocks_essay, blocks_update, blocks_essay_update

    Parameters:
        kwargs: Key-word arguments used to update the component's properties.
    Example:
        # Blocks Example
        import gradio as gr
        with gr.Blocks() as demo:
            radio = gr.Radio([1, 2, 4], label="Set the value of the number")
            number = gr.Number(value=2, interactive=True)
            radio.change(fn=lambda value: gr.update(value=value), inputs=radio, outputs=number)
        demo.launch()

        # Interface example
        import gradio as gr
        def change_textbox(choice):
          if choice == "short":
              return gr.Textbox.update(lines=2, visible=True)
          elif choice == "long":
              return gr.Textbox.update(lines=8, visible=True)
          else:
              return gr.Textbox.update(visible=False)
        gr.Interface(
          change_textbox,
          gr.Radio(
              ["short", "long", "none"], label="What kind of essay would you like to write?"
          ),
          gr.Textbox(lines=2),
          live=True,
        ).launch()
    """
    kwargs["__type__"] = "generic_update"
    return kwargs
