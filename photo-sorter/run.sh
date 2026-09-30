#!/bin/zsh
set -e
cd "${0:A:h}"
./build.sh
open -n "$PWD/.build/PhotoSorter.app"
