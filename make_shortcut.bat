@echo off
REM Creates "Arborist" shortcuts on the Desktop and in the Start menu.
cd /d %~dp0
set "HERE=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {" ^
  "  $s = $ws.CreateShortcut((Join-Path $dir 'Arborist.lnk'));" ^
  "  $s.TargetPath = Join-Path $env:HERE 'run_Arborist.bat';" ^
  "  $s.WorkingDirectory = $env:HERE;" ^
  "  $s.IconLocation = (Join-Path $env:HERE 'arborist\assets\Arborist.ico') + ',0';" ^
  "  $s.WindowStyle = 7;" ^
  "  $s.Description = 'Arborist - morphology of SC neurons';" ^
  "  $s.Save();" ^
  "  $old = Join-Path $dir 'SC_morph.lnk'; if (Test-Path $old) { Remove-Item $old } }"
if errorlevel 1 ( echo Could not create the shortcut. ) else ( echo Shortcut "Arborist" created on the Desktop and in the Start menu. )
if /i not "%~1"=="nopause" pause
