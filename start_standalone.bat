@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

:: ========================================================
:: 1. Config File Generation and Editing
:: ========================================================
set "CONFIG_FILE=config.txt"

:: Check if config file exists
if exist "%CONFIG_FILE%" (
    echo [INFO] Config file found. Skipping editor.
    :: If file exists, skip creation and go to parser
    goto :ConfigParse
)

:: --- Below runs only if config file is not found ---

echo [INFO] Config file not found. Creating template...

:: Create template
(
    echo PYTHON_VERSION=3.12.11
    echo CUDA_VERSION=cu130
) > "%CONFIG_FILE%"

echo [INFO] Opening Notepad. Please edit and save/close to continue...

:: Open notepad and wait until closed
::start /wait notepad.exe "%CONFIG_FILE%"
::echo [INFO] Editor closed. Resuming script...

:: ========================================================
:: 2. Parse Config File
:: ========================================================
:ConfigParse
echo [INFO] Reading configuration...

set "CFG_PYTHON="
set "CFG_CUDA="

:: Loop to read config file
for /f "usebackq tokens=1,2 delims==" %%a in ("%CONFIG_FILE%") do (
    set "key=%%a"
    set "val=%%b"
    
    :: Check keys
    if /i "!key!"=="PYTHON_VERSION" set "CFG_PYTHON=!val!"
    if /i "!key!"=="CUDA_VERSION" set "CFG_CUDA=!val!"
)

:: Check if values were retrieved
if "!CFG_PYTHON!"=="" (
    echo [ERROR] PYTHON_VERSION is missing in config.txt
    pause
    exit /b 1
)
if "!CFG_CUDA!"=="" (
    echo [ERROR] CUDA_VERSION is missing in config.txt
    pause
    exit /b 1
)

echo [CONFIG] Python Version: !CFG_PYTHON!
echo [CONFIG] CUDA Version  : !CFG_CUDA!

:: Validate CUDA version and define URL
set "TORCH_INDEX_URL="
set "IS_VALID_CUDA=0"

if "!CFG_CUDA!"=="cu126" (
    set "IS_VALID_CUDA=1"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126"
)
if "!CFG_CUDA!"=="cu128" (
    set "IS_VALID_CUDA=1"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
)
if "!CFG_CUDA!"=="cu130" (
    set "IS_VALID_CUDA=1"
    set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"
)

if "!IS_VALID_CUDA!"=="0" (
    echo [ERROR] Invalid CUDA_VERSION: !CFG_CUDA!
    echo [ERROR] Allowed values: cu126, cu128, cu130
    pause
    exit /b 1
)

:: ========================================================
:: 3. Git Environment Setup
:: ========================================================
echo [CHECK] Checking Git...

where git >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Git is already installed globally.
    goto :GitReady
)

set "MINGIT_DIR=.mingit_tmp"
if exist "%MINGIT_DIR%\cmd\git.exe" (
    echo [INFO] Found local MinGit in %MINGIT_DIR%.
    set "PATH=%~dp0%MINGIT_DIR%\cmd;%PATH%"
    goto :GitReady
)

echo [INFO] Git not found. Starting portable setup...

set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"

set "GIT_VER=v2.51.2.windows.1"
set "BASE_URL=https://github.com/git-for-windows/git/releases/download/%GIT_VER%"
set "ZIP_NAME="

if /i "%ARCH%"=="AMD64" (
    set "ZIP_NAME=MinGit-2.51.2-64-bit.zip"
) else if /i "%ARCH%"=="ARM64" (
    set "ZIP_NAME=MinGit-2.51.2-arm64.zip"
) else (
    set "ZIP_NAME=MinGit-2.51.2-32-bit.zip"
)

if not exist "%MINGIT_DIR%" mkdir "%MINGIT_DIR%"

echo [DOWNLOAD] Downloading %ZIP_NAME%...
curl -L -o "%MINGIT_DIR%\%ZIP_NAME%" "%BASE_URL%/%ZIP_NAME%"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git download failed.
    pause
    exit /b 1
)

echo [EXTRACT] Extracting MinGit...
tar -xf "%MINGIT_DIR%\%ZIP_NAME%" -C "%MINGIT_DIR%"

echo [SETUP] Setting temporary PATH for Git...
set "PATH=%~dp0%MINGIT_DIR%\cmd;%PATH%"

:GitReady
echo ---------------------------------------------------
git --version
echo ---------------------------------------------------

:: ========================================================
:: 4. uv Environment Setup
:: ========================================================
echo [CHECK] Checking uv...

where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] uv is already installed.
    goto :UvReady
)

set "UV_LOCAL_DIR=uv"
if exist "%UV_LOCAL_DIR%\uv.exe" (
    echo [INFO] Found local uv in %UV_LOCAL_DIR%.
    set "PATH=%~dp0%UV_LOCAL_DIR%;%PATH%"
    goto :UvReady
)

echo [INFO] uv not found. Installing portable uv...
powershell -ExecutionPolicy ByPass -Command "$env:UV_INSTALL_DIR = \"$PWD\%UV_LOCAL_DIR%\"; irm 'https://astral.sh/uv/install.ps1' | iex"
set "PATH=%~dp0%UV_LOCAL_DIR%;%PATH%"

:UvReady
echo ---------------------------------------------------
uv --version
echo ---------------------------------------------------

:: ========================================================
:: 5. Python Virtual Environment Setup
:: ========================================================
echo [CHECK] Checking Python Virtual Environment .venv...

if exist ".venv\Scripts\python.exe" (
    echo [INFO] .venv already exists. Skipping creation.
) else (
    echo [SETUP] Creating .venv with Python !CFG_PYTHON!...
    uv venv .venv --python !CFG_PYTHON!
)

echo [UPDATE] Installing/Updating PyTorch for !CFG_CUDA!...
echo [INFO] Index URL: !TORCH_INDEX_URL!

:: Execute dependency resolution and installation
uv pip install -p .venv -U pip setuptools wheel
uv pip install -p .venv torch torchvision --index-url !TORCH_INDEX_URL!

echo ---------------------------------------------------
echo [SUCCESS] Environment setup complete.
echo Python: !CFG_PYTHON!
echo CUDA  : !CFG_CUDA!
.venv\Scripts\python --version
:: Simple check for Torch version and CUDA availability
.venv\Scripts\python -c "import torch; print('Torch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
echo ---------------------------------------------------

uv pip install -p .venv -r requirements.txt

.venv\Scripts\python app.py

pause