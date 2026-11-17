#!/bin/bash
# Arborist installer / updater for macOS.
# Double-click in Finder. The first time macOS may block it: right-click > Open > Open.
# The program is copied to ~/Library/Application Support/Arborist, because apps may not read
# Downloads/Documents/Desktop (macOS privacy protection). Run again after downloading a new version.
cd "$(dirname "$0")" || exit 1
SRC="$(pwd)"
INSTALL_DIR="$HOME/Library/Application Support/Arborist"

fail() { echo; echo "$1"; read -n1 -r -p "Press any key to close"; exit 1; }

find_conda() {
  for d in "$HOME/miniconda3" "$HOME/opt/miniconda3" "$HOME/anaconda3" "$HOME/opt/anaconda3" \
           "$HOME/miniforge3" "/opt/miniconda3" "/opt/anaconda3" \
           "/opt/homebrew/Caskroom/miniconda/base" "/usr/local/Caskroom/miniconda/base" \
           "/opt/homebrew/Caskroom/miniforge/base"; do
    if [ -f "$d/etc/profile.d/conda.sh" ]; then echo "$d"; return; fi
  done
  if command -v conda >/dev/null 2>&1; then conda info --base 2>/dev/null; fi
}

# 1) copy the program out of the protected folder
mkdir -p "$INSTALL_DIR" || fail "Could not create $INSTALL_DIR"
rm -rf "$INSTALL_DIR/arborist"
cp -R "$SRC/arborist" "$SRC/requirements.txt" "$SRC/README.md" "$SRC/make_mac_app.sh" "$INSTALL_DIR/" \
  || fail "Copying the program failed."
find "$INSTALL_DIR" -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null
xattr -dr com.apple.quarantine "$INSTALL_DIR" 2>/dev/null
xattr -dr com.apple.quarantine "$SRC" 2>/dev/null

# 2) Python environment (reused if it already exists)
CONDA_ROOT="$(find_conda)"
if [ -n "$CONDA_ROOT" ]; then
  PY="$CONDA_ROOT/envs/Arborist/bin/python"
  # reuse the environment made when the program was still called SC_morph
  [ ! -x "$PY" ] && [ -x "$CONDA_ROOT/envs/SC_morph/bin/python" ] && PY="$CONDA_ROOT/envs/SC_morph/bin/python"
  if [ ! -x "$PY" ]; then
    echo "Found conda in $CONDA_ROOT - creating the Arborist environment"
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda create -y -n Arborist python=3.12 || fail "Creating the conda environment failed."
  else
    echo "Using the existing conda environment $(basename "$(dirname "$(dirname "$PY")")")"
  fi
else
  PY="$INSTALL_DIR/venv/bin/python"
  OLD_VENV="$HOME/Library/Application Support/SC_morph/venv"
  if [ ! -x "$PY" ] && [ -x "$OLD_VENV/bin/python" ]; then
    mv "$OLD_VENV" "$INSTALL_DIR/venv" 2>/dev/null && echo "Reusing the Python environment from SC_morph"
  fi
  if [ ! -x "$PY" ]; then
    SYSPY="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)"
    [ -z "$SYSPY" ] && fail "Neither conda nor Python 3 was found. Install Miniconda or Python 3.12 (python.org), then run install.command again."
    "$SYSPY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || \
      fail "$SYSPY is older than 3.10. Install Miniconda or Python 3.12 from python.org, then run install.command again."
    echo "Using $SYSPY - creating a Python environment in $INSTALL_DIR/venv"
    "$SYSPY" -m venv "$INSTALL_DIR/venv" || fail "Creating the environment failed."
  fi
fi
[ -x "$PY" ] || fail "Python not found at $PY"
echo "$PY" > "$INSTALL_DIR/python_path"

# 3) packages (fast when already installed)
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$INSTALL_DIR/requirements.txt" || fail "Package installation failed - see the messages above."

# 4) the app (the old SC_morph app and its Desktop link are removed)
rm -rf "$HOME/Applications/SC_morph.app" 2>/dev/null
[ -L "$HOME/Desktop/SC_morph" ] && rm -f "$HOME/Desktop/SC_morph"
bash "$INSTALL_DIR/make_mac_app.sh" || fail "Creating the app failed."
echo
echo "Done. Start Arborist from the Arborist app (Launchpad, ~/Applications, or the Desktop link)."
echo "You can delete the downloaded folder now; to update, run install.command of the new version."
read -n1 -r -p "Press any key to close"
