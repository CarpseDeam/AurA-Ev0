# src/aura/ui/banner.py

# No more 'rich' import! This is pure Python.

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
    Generates a self-contained HTML string for the AURA banner by manually
    building the styles for each character. This gives us total control.
    """
    lines = AURA_ART.strip('\n').split('\n')
    max_len = max(len(line) for line in lines) if lines else 0

    start_color_hex = "#00CED1"  # Cyan
    end_color_hex = "#A060DD"  # Purple

    start_r, start_g, start_b = _hex_to_rgb(start_color_hex)
    end_r, end_g, end_b = _hex_to_rgb(end_color_hex)

    # We will build the HTML line by line
    html_lines = []

    for line in lines:
        line_html = ""
        # IMPORTANT: Iterate up to max_len to handle padding for alignment
        for i, char in enumerate(line.ljust(max_len)):
            # Only color non-whitespace characters
            if char.strip():
                ratio = i / max(1, max_len - 1)

                # Interpolate the color
                r = int(start_r * (1 - ratio) + end_r * ratio)
                g = int(start_g * (1 - ratio) + end_g * ratio)
                b = int(start_b * (1 - ratio) + end_b * ratio)

                color_hex = _rgb_to_hex(r, g, b)

                # Each colored character is a <span> tag
                line_html += f'<span style="color: {color_hex};">{char}</span>'
            else:
                # Add spaces as-is
                line_html += " "
        html_lines.append(line_html)

    # Join all our styled lines with <br> tags
    inner_html = "<br>".join(html_lines)

    # Wrap the entire thing in a SINGLE <pre> tag with our bulletproof styling.
    # This is the secret sauce for perfect alignment.
    return (
        f'<pre style="font-family: \'JetBrains Mono\', \'Consolas\', monospace; '
        'font-size: 16px; font-weight: bold; line-height: 1.0; margin: 20px 0;">'
        f'{inner_html}'
        '</pre>'
    )