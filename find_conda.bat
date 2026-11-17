@echo off
REM Sets CONDA_ROOT to an existing Miniconda/Anaconda install (empty if none found).
set "CONDA_ROOT="
if defined CONDA_EXE for %%I in ("%CONDA_EXE%\..\..") do if exist "%%~fI\condabin\conda.bat" set "CONDA_ROOT=%%~fI"
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\Miniconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\miniconda3" "%ProgramData%\Miniconda3" "C:\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\anaconda3" "C:\anaconda3") do (
  if not defined CONDA_ROOT if exist "%%~D\condabin\conda.bat" set "CONDA_ROOT=%%~D"
)
if not defined CONDA_ROOT (
  for /f "delims=" %%P in ('where conda.bat 2^>nul') do if not defined CONDA_ROOT for %%I in ("%%~dpP..") do set "CONDA_ROOT=%%~fI"
)
