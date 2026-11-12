@echo off
setlocal

title Obsidian Voice Clone Studio (Chatterbox-TTS)

REM Navigate to the project directory regardless of where this .bat file is placed (e.g., Desktop)
cd /d "c:\Users\Thedr\OneDrive\Desktop\Voice Clone app"

REM Locate and activate the Python virtual environment
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo [ERROR] No Python virtual environment found ^(.venv or venv^) in:
    echo   %CD%
    echo.
    echo Run setup_env.bat in that folder first, or create .venv manually:
    echo   uv venv .venv --python 3.11
    echo   .venv\Scripts\activate.bat
    echo   pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Optimize PyTorch CUDA allocator for 6 GB VRAM on Windows (Lenovo LOQ RTX 3050)
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set GRADIO_ANALYTICS_ENABLED=False

echo Starting Obsidian Voice Clone Studio on http://localhost:7860 ...

REM Poll localhost:7860 in background (up to 120s), then open the default web browser
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "$deadline = (Get-Date).AddSeconds(120); while ((Get-Date) -lt $deadline) { try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 7860); $c.Close(); Start-Process 'http://localhost:7860'; break } catch { Start-Sleep -Milliseconds 750 } }"

REM Launch app.py (--no-browser avoids opening a duplicate tab)
python app.py --no-browser

pause
