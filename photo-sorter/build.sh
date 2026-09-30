#!/bin/zsh
set -e
cd "${0:A:h}"

typeset executable_path=""
if swift package --version >/dev/null 2>&1; then
  swift build
  executable_path=".build/debug/PhotoSorter"
else
  # A mismatched Command Line Tools installation can break Swift Package Manager
  # while swiftc itself still works. Choose the newest locally installed SDK that
  # the compiler accepts, then build the executable directly.
  mkdir -p .build/manual
  typeset selected_sdk=""
  typeset target_arch="$(uname -m)"
  for sdk in ${(On)${(f)"$(find /Library/Developer/CommandLineTools/SDKs -maxdepth 1 -type d -name 'MacOSX*.sdk' 2>/dev/null)"}}; do
    if printf 'import Foundation\n' | swiftc -typecheck -sdk "$sdk" -target "${target_arch}-apple-macosx14.0" - >/dev/null 2>&1; then
      selected_sdk="$sdk"
      break
    fi
  done

  if [[ -z "$selected_sdk" ]]; then
    print -u2 "找不到与当前 Swift 编译器匹配的 macOS SDK。请更新 Xcode Command Line Tools。"
    exit 1
  fi

  swiftc -parse-as-library -O \
    -sdk "$selected_sdk" -target "${target_arch}-apple-macosx14.0" \
    Sources/PhotoSorter/**/*.swift \
    -framework SwiftUI -framework AppKit -framework ImageIO -framework QuickLookThumbnailing \
    -o .build/manual/PhotoSorter
  executable_path=".build/manual/PhotoSorter"
fi

typeset app_path=".build/PhotoSorter.app"
mkdir -p "$app_path/Contents/MacOS" "$app_path/Contents/Resources"
cp "$executable_path" "$app_path/Contents/MacOS/PhotoSorter"
cp Resources/Info.plist "$app_path/Contents/Info.plist"
chmod +x "$app_path/Contents/MacOS/PhotoSorter"
codesign --force --sign - "$app_path" >/dev/null 2>&1 || true
print "已构建：${PWD}/${app_path}"
