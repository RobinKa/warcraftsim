#!/bin/bash
# GE-Proton's Wine (10.0 with fsync) for the games: 28% less CPU a step than WineHQ stable 11.0 with
# WINEFSYNC=1 (selfplay --wine ~/wc3/ge/GE-Proton10-34/files/bin --fsync 1). Proton 11 builds need
# glibc 2.38 (Ubuntu 22.04 has 2.35); 10-34 runs here. Outside Steam: runtime.wine sets the library
# paths its proton script would, and copies vkd3d into the template prefix.
set -euo pipefail
TAG=${1:-GE-Proton10-34}
DEST=${GE_DIR:-$HOME/wc3/ge}
mkdir -p "$DEST" && cd "$DEST"
[ -d "$TAG" ] && { echo "$DEST/$TAG/files/bin"; exit 0; }
base=https://github.com/GloriousEggroll/proton-ge-custom/releases/download/$TAG
curl -sSL -o "$TAG.sha512sum" "$base/$TAG.sha512sum"
curl -sSL -o "$TAG.tar.gz" "$base/$TAG.tar.gz"
sha512sum -c "$TAG.sha512sum"
tar xzf "$TAG.tar.gz" && rm "$TAG.tar.gz"
echo "$DEST/$TAG/files/bin"
