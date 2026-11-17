@echo off
cd /d "%~dp0"
set "HERE=%~dp0"
set "PYTHONPATH=%~dp0"
if exist .venv\Scripts\activate.bat goto venv
call find_conda.bat
if not defined CONDA_ROOT goto missing
set "ENVNAME=Arborist"
if not exist "%CONDA_ROOT%\envs\Arborist\python.exe" if exist "%CONDA_ROOT%\envs\SC_morph\python.exe" set "ENVNAME=SC_morph"
call "%CONDA_ROOT%\condabin\conda.bat" activate %ENVNAME%
if errorlevel 1 goto missing
goto start
:venv
call .venv\Scripts\activate.bat
:start
cd /d "%HERE%"
python -m arborist
if not errorlevel 1 exit /b
echo.
echo Arborist stopped with an error. Checking the installation...
python "%HERE%arborist\check_env.py" --repair
if errorlevel 10 goto again
pause
exit /b
:again
echo Starting Arborist again...
python -m arborist
if errorlevel 1 pause
exit /b
:missing
echo Arborist is not installed yet - run install.bat first.
pause
