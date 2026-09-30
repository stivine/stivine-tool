#!/bin/zsh
set -e
cd "${0:A:h}"
mkdir -p .build/manual
typeset target_arch="$(uname -m)"
typeset selected_sdk=""
for sdk in ${(On)${(f)"$(find /Library/Developer/CommandLineTools/SDKs -maxdepth 1 -type d -name 'MacOSX*.sdk' 2>/dev/null)"}}; do
  if printf 'import Foundation\n' | swiftc -typecheck -sdk "$sdk" -target "${target_arch}-apple-macosx14.0" - >/dev/null 2>&1; then
    selected_sdk="$sdk"
    break
  fi
done
[[ -n "$selected_sdk" ]] || { print -u2 "找不到兼容的 macOS SDK"; exit 1; }

swiftc -parse-as-library -sdk "$selected_sdk" -target "${target_arch}-apple-macosx14.0" \
  Sources/PhotoSorter/Models/*.swift \
  Sources/PhotoSorter/Services/*.swift \
  Validation/ValidationMain.swift \
  -framework SwiftUI -framework AppKit -framework ImageIO -framework QuickLookThumbnailing \
  -o .build/manual/PhotoSorterValidation
.build/manual/PhotoSorterValidation
