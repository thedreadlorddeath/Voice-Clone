@echo off
setlocal

cd /d "c:\Users\Thedr\OneDrive\Desktop\Voice Clone app"

set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if exist "%PY311%" (
    echo Creating Python 3.11 virtual environment ^(.venv^) using official signed Python...
    "%PY311%" -m venv .venv
) else (
    echo Official Python 3.11 not found at "%PY311%".
    echo Installing Python 3.11 via winget ^(signed by Python Software Foundation^)...
    winget install --id Python.Python.3.11 --exact --scope user --silent --accept-package-agreements --accept-source-agreements --override "/quiet InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_test=0"
    if exist "%PY311%" (
        "%PY311%" -m venv .venv
    ) else (
        echo Falling back to uv venv .venv --python 3.11...
        uv venv .venv --python 3.11
    )
)

if errorlevel 1 (
    echo [ERROR] Failed to create .venv.
    pause
    exit /b 1
)

echo Installing CUDA-enabled PyTorch 2.6.0 ^(cu124^) for RTX 3050 6GB...
uv pip install --python .venv\Scripts\python.exe torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch CUDA packages.
    pause
    exit /b 1
)

echo Installing app dependencies from requirements.txt...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.txt dependencies.
    pause
    exit /b 1
)

echo.
echo Setup complete!
echo Place a clean 5-15s voice sample named my_voice.wav in this folder ^(or upload one in the UI^),
echo then double-click start_tts.bat ^(you can copy start_tts.bat to your Desktop^).
pause
