@echo off
rem Double-click this file in File Explorer to start the digit recognizer.
setlocal
cd /d "%~dp0"
title Digit Recognizer
echo === Digit Recognizer ===

rem 1. Find Python 3.10 or newer.
set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto have_python
python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto have_python
echo.
echo Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/
echo then double-click this file again.
goto fail

:have_python
set "VPY=.venv\Scripts\python.exe"

rem 2. Private Python environment for this app (made once; remade if it was copied from another computer).
"%VPY%" -c "import sys" >nul 2>nul
if not errorlevel 1 goto have_venv
echo Setting up for first use - one time only...
if exist .venv rmdir /s /q .venv
%PY% -m venv .venv
if errorlevel 1 goto fail
:have_venv

rem 3. Install packages (again only if requirements.txt changed).
fc /b requirements.txt .venv\installed.txt >nul 2>nul
if not errorlevel 1 goto have_packages
echo Installing packages...
"%VPY%" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo Package install failed. Check your internet connection and try again.
  goto fail
)
copy /y requirements.txt .venv\installed.txt >nul
:have_packages

rem 4. Train the model the first time (downloads MNIST, about 2 minutes).
if exist model.joblib goto have_model
echo Training the recognizer for the first time - about 2 minutes...
"%VPY%" train.py
if errorlevel 1 goto fail
:have_model

rem 5. Start the app and open the browser.
"%VPY%" app.py --open
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
pause
exit /b 1
