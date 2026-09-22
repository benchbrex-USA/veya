"""Write the Veya mark as SVG.

The PNG mark and wordmark in this directory are the supplied brand images:

  logo-mark.png, logo-mark-ink.png          icon on blue
  logo-lockup.png, logo-lockup-dark.png     icon plus the Veya wordmark

This script refreshes the SVG companions only. It does not redraw those PNGs.
"""
import os

ASSETS = os.path.dirname(os.path.abspath(__file__))
BLUE = "#020076"
MARK = """
    <rect x="244" y="160" width="13" height="55"/>
    <rect x="244" y="285" width="13" height="55"/>
    <rect x="162" y="246" width="58" height="12"/>
    <rect x="280" y="246" width="58" height="12"/>
    <circle cx="203" cy="201.5" r="8.5"/>
    <circle cx="293" cy="201.5" r="8.5"/>
    <circle cx="207.5" cy="297" r="8"/>
    <circle cx="293" cy="297" r="8.5"/>
"""


def icon_svg(fill, background):
    bg = '<rect width="500" height="500" fill="%s"/>' % background if background else ""
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="500" height="500" role="img" aria-label="Veya">
  <title>Veya</title>
  %s
  <g fill="%s">%s  </g>
</svg>
""" % (bg, fill, MARK)


def lockup_svg():
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="500" height="500" role="img" aria-label="Veya">
  <title>Veya</title>
  <rect width="500" height="500" fill="%s"/>
  <g transform="translate(101 208) scale(0.164)" fill="#ffffff">%s  </g>
  <text x="194" y="278" fill="#ffffff" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="72" font-weight="500" letter-spacing="-1">Veya</text>
</svg>
""" % (BLUE, MARK)


def main():
    files = {
        "logo-mark.svg": icon_svg("#ffffff", BLUE),
        "logo-mark-ink.svg": icon_svg("#ffffff", BLUE),
        "logo-mark-mono.svg": icon_svg("currentColor", None),
        "logo-lockup.svg": lockup_svg(),
        "logo-lockup-dark.svg": lockup_svg(),
    }
    for name, svg in files.items():
        path = os.path.join(ASSETS, name)
        with open(path, "w") as f:
            f.write(svg)
        print(name)


if __name__ == "__main__":
    main()
