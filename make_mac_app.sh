#!/bin/bash
# Creates ~/Applications/Arborist.app (with the logo) and a link to it on the Desktop.
INSTALL_DIR="$HOME/Library/Application Support/Arborist"
PY="$(cat "$INSTALL_DIR/python_path" 2>/dev/null)"
[ -x "$PY" ] || { echo "Run install.command first."; exit 1; }
APP="$HOME/Applications/Arborist.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$INSTALL_DIR/arborist/assets/Arborist.icns" "$APP/Contents/Resources/Arborist.icns"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Arborist</string>
  <key>CFBundleDisplayName</key><string>Arborist</string>
  <key>CFBundleIdentifier</key><string>hu.pte.arborist</string>
  <key>CFBundleExecutable</key><string>Arborist</string>
  <key>CFBundleIconFile</key><string>Arborist</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSRemovableVolumesUsageDescription</key><string>Arborist opens image stacks from your data drive.</string>
  <key>NSDocumentsFolderUsageDescription</key><string>Arborist opens image stacks you choose.</string>
  <key>NSDesktopFolderUsageDescription</key><string>Arborist opens image stacks you choose.</string>
  <key>NSDownloadsFolderUsageDescription</key><string>Arborist opens image stacks you choose.</string>
</dict></plist>
PLIST
cat > "$APP/Contents/MacOS/Arborist" <<LAUNCH
#!/bin/bash
# Starts Arborist from ~/Library/Application Support/Arborist; messages go to ~/Library/Logs/Arborist.log
mkdir -p "\$HOME/Library/Logs"
cd "$INSTALL_DIR" || exit 1
echo "--- \$(date) starting Arborist" >> "\$HOME/Library/Logs/Arborist.log"
LOG="\$HOME/Library/Logs/Arborist.log"
"$PY" -m arborist >> "\$LOG" 2>&1 && exit 0
echo "--- error: checking the installation" >> "\$LOG"
"$PY" -m arborist.check_env --repair >> "\$LOG" 2>&1
if [ \$? -eq 10 ]; then
  exec "$PY" -m arborist >> "\$LOG" 2>&1
fi
osascript -e 'display alert "Arborist could not start" message "Details are in ~/Library/Logs/Arborist.log. Running install.command again usually fixes it."' >/dev/null 2>&1
LAUNCH
chmod +x "$APP/Contents/MacOS/Arborist"
xattr -dr com.apple.quarantine "$APP" 2>/dev/null
touch "$APP"
ln -sfn "$APP" "$HOME/Desktop/Arborist"
echo "App created: $APP (and a link on the Desktop)"
