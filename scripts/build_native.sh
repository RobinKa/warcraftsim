#!/usr/bin/env bash
# Build native helpers into build/: StormLib (libstorm.so) and pjass.
# Needs no root: a recent cmake comes from the venv (pip install cmake), and flex/bison/m4
# are extracted locally from Ubuntu .debs when they are not installed system-wide.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
git submodule update --init third_party/StormLib third_party/pjass

CMAKE=cmake
if [[ -x .venv/bin/cmake ]]; then CMAKE=.venv/bin/cmake; fi

# --- StormLib ---
"$CMAKE" -S third_party/StormLib -B build/stormlib \
  -DBUILD_SHARED_LIBS=ON -DSTORM_USE_BUNDLED_LIBRARIES=ON -DSTORM_SKIP_INSTALL=ON \
  -DCMAKE_BUILD_TYPE=Release >/dev/null
"$CMAKE" --build build/stormlib -j "$(nproc)" >/dev/null
echo "built build/stormlib/libstorm.so"

# --- pjass (needs flex + bison) ---
if ! command -v flex >/dev/null || ! command -v bison >/dev/null; then
  mkdir -p build/localdeb
  (cd build/localdeb && apt-get download flex bison m4 libfl2 libfl-dev >/dev/null &&
   for f in *.deb; do dpkg -x "$f" root; done)
  L=$ROOT/build/localdeb/root
  export PATH=$L/usr/bin:$PATH M4=$L/usr/bin/m4 BISON_PKGDATADIR=$L/usr/share/bison
fi
rm -rf build/pjass
cp -r third_party/pjass build/pjass
make -C build/pjass -j "$(nproc)" pjass >/dev/null 2>&1
echo "built build/pjass/pjass"
