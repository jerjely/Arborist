#!/bin/bash
# Start Arborist from Terminal (macOS). Shows messages in this window.
INSTALL_DIR="$HOME/Library/Application Support/Arborist"
PY="$(cat "$INSTALL_DIR/python_path" 2>/dev/null)"
if [ ! -x "$PY" ]; then
  echo "Arborist is not installed yet - double-click install.command first."
  read -n1 -r -p "Press any key to close"; exit 1
fi
cd "$INSTALL_DIR" || exit 1
"$PY" -m arborist && exit 0
echo
echo "Arborist stopped with an error. Checking the installation..."
"$PY" -m arborist.check_env --repair
if [ $? -eq 10 ]; then
  echo "Starting Arborist again..."
  "$PY" -m arborist && exit 0
fi
read -n1 -r -p "Press any key to close"
