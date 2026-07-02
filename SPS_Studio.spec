# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 🔽 IMPORTS MINIMOS (evitar inflar el exe)
hiddenimports = [
    # Qt
    "PySide6",
    
    # Matplotlib backend Qt
    "matplotlib.backends.backend_qtagg",

    # SciPy necesario para cálculos
    "scipy.spatial._qhull",

    # DB
    "pyodbc",
]

# 🔽 SOLO recursos reales
datas = [
    ("img", "img"),
]

# 🔽 EXCLUSIONES (limpiar ruido)
excludes = [
    "tkinter",
    "pytest",
    "test",
    "IPython",
    "jupyter",
    "notebook",
    "setuptools",
    "distutils",

    # backends no usados
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends.backend_wxagg",
    "matplotlib.backends.backend_gtk3agg",
    "matplotlib.backends.backend_gtk4agg",
    "matplotlib.backends.backend_macosx",
    "matplotlib.backends.backend_webagg",
    "matplotlib.backends.backend_nbagg",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,

    # 🔥 CLAVE PARA DEFENDER
    noarchive=True,
)

pyz = PYZ(
    a.pure,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,

    # 🔥 IMPORTANTE en onedir
    a.binaries,
    a.zipfiles,
    a.datas,

    [],
    name="SPSStudio_v2.21.1",

    debug=False,
    bootloader_ignore_signals=False,
    strip=False,

    # 🔥 NO UPX
    upx=False,
    upx_exclude=[],

    console=False,
    disable_windowed_traceback=False,

    icon="img/SPS_icon.ico",

    # opcional pero recomendable
    version="version.txt",
)