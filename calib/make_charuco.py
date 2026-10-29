#!/usr/bin/env python3
"""Generate the ChArUco board on A4 landscape (print at 100 % scale). Must match calib_touch.py settings.

  python3 calib/make_charuco.py   ->  calib/charuco_7x5_35mm_A4.pdf / .png / _preview.png
Measure a printed square with a ruler and pass it as --square-mm to calib_touch.py.
"""
from pathlib import Path

import cv2
from PIL import Image

SQUARES_X, SQUARES_Y = 7, 5
SQUARE_MM, MARKER_MM = 35.0, 26.0
DICT = cv2.aruco.DICT_5X5_100
DPI = 600
A4_MM = (297.0, 210.0)


def mm2px(mm):
    return int(round(mm / 25.4 * DPI))


if __name__ == "__main__":
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM / 1000, MARKER_MM / 1000,
                                   cv2.aruco.getPredefinedDictionary(DICT))
    bw, bh = mm2px(SQUARES_X * SQUARE_MM), mm2px(SQUARES_Y * SQUARE_MM)
    img = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    page = Image.new("L", (mm2px(A4_MM[0]), mm2px(A4_MM[1])), 255)
    page.paste(Image.fromarray(img), ((page.width - bw) // 2, (page.height - bh) // 2))
    out = str(Path(__file__).parent / "charuco_7x5_35mm_A4")
    page.save(out + ".pdf", resolution=DPI)
    page.save(out + ".png", dpi=(DPI, DPI))
    page.resize((page.width // 6, page.height // 6)).save(out + "_preview.png")
    print(out + ".pdf", page.size)
