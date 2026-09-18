$ErrorActionPreference = "Stop"

.\.venv\Scripts\python.exe -m pip install pyinstaller
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean PneumoniaPredictor.spec

Write-Host ""
Write-Host "Build finished:"
Write-Host "dist\PneumoniaPredictor\PneumoniaPredictor.exe"
