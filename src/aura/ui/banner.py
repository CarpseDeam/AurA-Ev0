# src/aura/ui/banner.py

from rich.console import Console
from rich.style import Style
from rich.text import Text
from rich.theme import Theme

AURA_ART = """
  █████╗ ██╗   ██╗██████╗  █████╗ 
 ██╔══██╗██║   ██║██╔══██╗██╔══██╗
 ███████║██║   ██║██████╔╝███████║
 ██╔══██║██║   ██║██╔══██╗██╔══██║
 ██║  ██║╚██████╔╝██║  ██║██║  ██║
 ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
"""


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Converts a hex color string to an (r, g, b) tuple."""
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Converts an (r, g, b) tuple to a hex color string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def generate_banner_html() -> str:
    """
    Generates a self-contained HTML string for the AURA banner with a smooth
    color gradient, ready to be displayed in a Qt widget.
    """
    lines = AURA_ART.strip('\n').split('\n')
    max_len = max(len(line) for line in lines) if lines else 0

    start_color_hex = "#00CED1"  # A nice, bright cyan
    end_color_hex = "#A060DD"  # A beautiful magenta/purple

    start_r, start_g, start_b = _hex_to_rgb(start_color_hex)
    end_r, end_g, end_b = _hex_to_rgb(end_color_hex)

    # Use Rich's Text object to build the styled text in memory
    rich_text = Text()

    for line in lines:
        padded_line = line.ljust(max_len)
        for i, char in enumerate(padded_line):
            if char.strip():
                # Calculate the ratio for linear interpolation
                # Use max(1, max_len - 1) to avoid division by zero
                ratio = i / max(1, max_len - 1)

                # Interpolate each color component
                r = int(start_r * (1 - ratio) + end_r * ratio)
                g = int(start_g * (1 - ratio) + end_g * ratio)
                b = int(start_b * (1 - ratio) + end_b * ratio)

                color_hex = _rgb_to_hex(r, g, b)
                rich_text.append(char, style=Style(color=color_hex, bold=True))
            else:
                rich_text.append(char)  # Append whitespace without style
        rich_text.append('\n')

    # Create a console with a minimal theme to prevent default styles
    # from interfering with our gradient.
    minimal_theme = Theme({"default": Style()})
    console = Console(record=True, width=max_len, theme=minimal_theme)

    console.print(rich_text)

    # Export the recorded content to HTML with inline styles
    html_content = console.export_html(inline_styles=True)

    # **THE FIX!** Wrap the exported HTML in our own <pre> tag with guaranteed
    # styling for the Aura GUI, ensuring perfect font and alignment.
    return (
        '<pre style="font-family: \'JetBrains Mono\', \'Consolas\', monospace; '
        'font-size: 16px; line-height: 0.95; margin: 20px 0; color: #e6edf3;">'
        f'{html_content}'
        '</pre>'
    )