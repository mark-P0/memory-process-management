"""
To be built with PyInstaller

Use the following command
    pyinstaller main.spec

Output folder
    `dist`
"""

# -*- mode: python ; coding: utf-8 -*-

# import sys
# import os
from kivy_deps import sdl2, glew
from kivymd import hooks_path as kivymd_hooks_path
from pathlib import Path


path = Path(".").resolve()  # Current working directory


names = {
    "executable": "06-ProcMgmt-SJF",
    "icon": str(path / "graphics" / "icon.ico"),
    "entrypoint": str(path / "main.py"),
    "with-prompt": False,
}

if names["with-prompt"]:
    names["executable"] += "-DEBUG"


# fmt: off
exclusions = f"""
.venv
.exclude
__pycache__
dist
build
layout.pptx
desktop.ini
main.spec
""".strip().split("\n")
# fmt: on


data_files = [
    (
        str(item),
        item.name if item.is_dir() else ".",
    )
    for item in path.iterdir()
    if item.name not in exclusions
]
print(*data_files, sep="\n")


a = Analysis(
    [names["entrypoint"]],
    pathex=[str(path)],
    datas=data_files,
    hiddenimports=[
        "kivymd.effects.stiffscroll.StiffScrollEffect",
        "kivymd.effects.stiffscroll",
        "kivymd.effects",
    ],
    hookspath=[kivymd_hooks_path],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
    debug=False,
    strip=False,
    upx=True,
    name=names["executable"],
    console=names["with-prompt"],
    icon=names["icon"],
)
