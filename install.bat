@echo off
setlocal EnableDelayedExpansion
REM One-time setup for Arborist.
REM Uses Miniconda/Anaconda if installed (creates a conda env "Arborist" - or reuses the
REM "SC_morph" env from before the program was renamed),
REM otherwise a plain Python install (creates .venv in this folder).
cd /d "%~dp0"
set "HERE=%~dp0"
set "PYTHONPATH=%~dp0"
call find_conda.bat

if defined CONDA_ROOT (
  echo Found conda in %CONDA_ROOT%
  set "ENVNAME=Arborist"
  if not exist "%CONDA_ROOT%\envs\Arborist\python.exe" if exist "%CONDA_ROOT%\envs\SC_morph\python.exe" set "ENVNAME=SC_morph"
  if not exist "%CONDA_ROOT%\envs\!ENVNAME!\python.exe" (
    call "%CONDA_ROOT%\condabin\conda.bat" create -y -n Arborist python=3.12
    if errorlevel 1 ( echo Creating the conda environment failed. & pause & exit /b 1 )
  ) else (
    echo Reusing the existing environment !ENVNAME!
  )
  call "%CONDA_ROOT%\condabin\conda.bat" activate !ENVNAME!
  cd /d "%HERE%"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python "%HERE%arborist\check_env.py" --repair --install
  if errorlevel 1 ( echo Package installation failed - see the messages above. & pause & exit /b 1 )
  call make_shortcut.bat nopause
  echo.
  echo Done. Start Arborist from the desktop shortcut ^(or run_Arborist.bat^).
  pause
  exit /b 0
)

REM ---- no conda: plain Python ----
set "PY="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if not defined PY (
  python --version >nul 2>&1
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo Neither conda nor Python was found - installing Python 3.12 with winget ...
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
if not defined PY (
  echo.
  echo Could not find conda or Python. Install Miniconda or Python 3.12 and run install.bat again.
  pause
  exit /b 1
)
echo Using: %PY%
%PY% -m venv .venv
if errorlevel 1 ( echo Creating the environment failed. & pause & exit /b 1 )
call .venv\Scripts\activate
cd /d "%HERE%"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python "%HERE%arborist\check_env.py" --repair --install
if errorlevel 1 ( echo Package installation failed - see the messages above. & pause & exit /b 1 )
call make_shortcut.bat nopause
echo.
echo Done. Start Arborist from the desktop shortcut ^(or run_Arborist.bat^).
pause
