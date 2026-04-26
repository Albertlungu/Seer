#!/bin/bash
set -e

echo "Building Seer.app locally..."

# Clean previous build
rm -rf python pbs.tar.gz Seer.app dmg-stage Seer-macOS.dmg

# Fetch self-contained Python (no external framework dependencies)
curl -L "https://github.com/indygreg/python-build-standalone/releases/download/20250127/cpython-3.12.9+20250127-aarch64-apple-darwin-install_only.tar.gz" -o pbs.tar.gz
tar -xf pbs.tar.gz   # extracts to python/
rm pbs.tar.gz

python/bin/pip3 install --upgrade pip -q
python/bin/pip3 install -r requirements-prod.txt -q

# Assemble .app bundle
mkdir -p Seer.app/Contents/MacOS
mkdir -p Seer.app/Contents/Resources

cat > Seer.app/Contents/MacOS/Seer << 'EOF'
#!/bin/bash
RESOURCES="$(cd "$(dirname "$0")/../Resources" && pwd)"
exec "$RESOURCES/python-dist/bin/python3" "$RESOURCES/main.py"
EOF
chmod +x Seer.app/Contents/MacOS/Seer

cat > Seer.app/Contents/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Seer</string>
  <key>CFBundleExecutable</key><string>Seer</string>
  <key>CFBundleIdentifier</key><string>com.seer.molecular</string>
  <key>CFBundleVersion</key><string>1.0.0</string>
  <key>CFBundleShortVersionString</key><string>1.0.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
</dict>
</plist>
EOF

cp assets/seer-icon.icns Seer.app/Contents/Resources/AppIcon.icns
cp main.py Seer.app/Contents/Resources/
cp -r src Seer.app/Contents/Resources/
cp -r data Seer.app/Contents/Resources/
cp -r python Seer.app/Contents/Resources/python-dist

echo "Seer.app assembled."

# Optionally wrap in DMG
if command -v create-dmg &> /dev/null; then
    mkdir dmg-stage
    cp -r Seer.app dmg-stage/Seer.app
    create-dmg \
        --volname "Seer" \
        --background "assets/dmg-background.png" \
        --window-size 1097 844 \
        --icon-size 160 \
        --icon "Seer.app" 274 422 \
        --app-drop-link 823 422 \
        "Seer-macOS.dmg" \
        dmg-stage
    echo "Seer-macOS.dmg created."
else
    echo "create-dmg not found — skipping DMG. Install with: brew install create-dmg"
fi

echo "Done. To install: cp -r Seer.app /Applications/Seer.app"
