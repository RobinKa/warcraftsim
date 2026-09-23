#!/usr/bin/env bash
# System packages for warcraftsim (Ubuntu 22.04 / WSL2). Run with sudo:
#   sudo bash scripts/setup_system.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run as root: sudo bash $0" >&2
  exit 1
fi

. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-$VERSION_CODENAME}"

# 32-bit userland: Warcraft III 1.29 is a 32-bit executable.
dpkg --add-architecture i386

# WineHQ repository (Ubuntu's own wine 6.0 is too old).
mkdir -pm755 /etc/apt/keyrings
wget -qO /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
wget -qNP /etc/apt/sources.list.d/ "https://dl.winehq.org/wine-builds/ubuntu/dists/${CODENAME}/winehq-${CODENAME}.sources"

# Unrelated broken third-party repos (expired keys, dead URLs) make `apt-get update` exit
# non-zero; only the WineHQ and Ubuntu indexes matter here, so keep going.
apt-get update || echo "warning: apt-get update reported errors (continuing)" >&2
DEBIAN_FRONTEND=noninteractive apt-get install -y --install-recommends winehq-stable
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  xvfb x11-utils x11-apps xdotool imagemagick mesa-utils \
  libgl1:i386 libgl1-mesa-dri:i386 libglx-mesa0:i386 \
  cabextract winbind \
  fuse3 libfuse3-dev pkg-config \
  flex bison build-essential cmake \
  mingw-w64 \
  python3-dev python3-venv \
  lua5.3

# Let non-root users mount FUSE filesystems with allow_other (IPC mounts inside Wine prefixes).
if ! grep -q '^user_allow_other' /etc/fuse.conf; then
  echo user_allow_other >> /etc/fuse.conf
fi

wine --version
echo "system setup done"
