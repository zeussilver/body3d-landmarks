"""Screenshot + GIF of the viewer for the README, using a local headless Chrome.

    python -m http.server 8790 --directory viewer &      # serve the viewer
    .venv/bin/python viewer/capture.py                  # -> docs/img/viewer.png, viewer.gif
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
URL = "http://localhost:8790/?subject={subject}&week={week}&ghost=1"
OUT = Path(__file__).parents[1] / "docs" / "img"
SIZE = (1280, 860)


def shot(url: str, path: Path, profile: str) -> None:
    subprocess.run([CHROME, "--headless=new", f"--user-data-dir={profile}", "--no-first-run",
                    "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars",
                    f"--window-size={SIZE[0]},{SIZE[1]}", "--virtual-time-budget=20000",
                    f"--screenshot={path}", url], check=True, capture_output=True)


def main(subject: str = "s5") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        frames = []
        for week in (0, 8, 16, 24):
            p = Path(tmp) / f"w{week}.png"
            shot(URL.format(subject=subject, week=week), p, f"{tmp}/profile")
            frames.append(Image.open(p).convert("RGB"))
        frames[-1].save(OUT / "viewer.png", optimize=True)
        small = [f.resize((SIZE[0] * 3 // 4, SIZE[1] * 3 // 4), Image.LANCZOS).quantize(colors=128) for f in frames]
        small[0].save(OUT / "viewer.gif", save_all=True, append_images=small[1:], duration=[900, 900, 900, 1800], loop=0)
    print("wrote", OUT / "viewer.png", OUT / "viewer.gif")


if __name__ == "__main__":
    main(*sys.argv[1:])
