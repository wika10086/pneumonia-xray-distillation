# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.building.datastruct import TOC
from PyInstaller.utils.hooks import collect_all


project_dir = Path.cwd()
blocked_runtime_dlls = {
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
}

datas = [
    (str(project_dir / "checkpoints" / "distilled_student_augmented_best.pt"), "checkpoints"),
]
binaries = []
hiddenimports = []

for package_name in ("torch", "torchvision", "PIL"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports


a = Analysis(
    ["app_gui.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "matplotlib",
        "pandas",
        "pytest",
        "scipy",
        "sklearn",
        "torchaudio",
    ],
    noarchive=False,
    optimize=0,
)


def without_blocked_runtime_dlls(toc):
    kept = []
    for entry in toc:
        dest_name = Path(entry[0]).name.lower()
        source_name = Path(entry[1]).name.lower()
        if dest_name in blocked_runtime_dlls or source_name in blocked_runtime_dlls:
            continue
        kept.append(entry)
    return TOC(kept)


a.binaries = without_blocked_runtime_dlls(a.binaries)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PneumoniaPredictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PneumoniaPredictor",
)
