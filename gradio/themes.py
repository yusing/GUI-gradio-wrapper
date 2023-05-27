class Theme:
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Parameters:
            primary_hue: The primary hue of the theme. Load a preset, like gradio.themes.colors.green (or just the string "green"), or pass your own gradio.themes.utils.Color object.
            secondary_hue: The secondary hue of the theme. Load a preset, like gradio.themes.colors.green (or just the string "green"), or pass your own gradio.themes.utils.Color object.
            neutral_hue: The neutral hue of the theme, used . Load a preset, like gradio.themes.colors.green (or just the string "green"), or pass your own gradio.themes.utils.Color object.
            text_size: The size of the text. Load a preset, like gradio.themes.sizes.text_sm (or just the string "sm"), or pass your own gradio.themes.utils.Size object.
            spacing_size: The size of the spacing. Load a preset, like gradio.themes.sizes.spacing_sm (or just the string "sm"), or pass your own gradio.themes.utils.Size object.
            radius_size: The radius size of corners. Load a preset, like gradio.themes.sizes.radius_sm (or just the string "sm"), or pass your own gradio.themes.utils.Size object.
            font: The primary font to use for the theme. Pass a string for a system font, or a gradio.themes.font.GoogleFont object to load a font from Google Fonts. Pass a list of fonts for fallbacks.
            font_mono: The monospace font to use for the theme, applies to code. Pass a string for a system font, or a gradio.themes.font.GoogleFont object to load a font from Google Fonts. Pass a list of fonts for fallbacks.
        """
        pass


Base = Theme
Default = Theme
