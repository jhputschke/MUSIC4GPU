#!/usr/bin/env bash

###############################################################################
# Copyright (c) The JETSCAPE Collaboration, 2018
#
# For the list of contributors see AUTHORS.
#
# Report issues at https://github.com/JETSCAPE/JETSCAPE/issues
#
# or via email to bugs.jetscape@gmail.com
#
# Distributed under the GNU General Public License 3.0 (GPLv3 or later).
# See COPYING for details.
##############################################################################
#
# Download Kokkos for the music4gpu USE_KOKKOS backend.
#
#   bash get_kokkos.sh            # latest release (default)
#   bash get_kokkos.sh 5.1.1      # a specific, pinned release tag
#
# Works both stand-alone (run from the music4gpu repo) and inside X-SCAPE
# (external_packages/music4gpu/get_kokkos.sh).  Kokkos is cloned into
# external/kokkos next to THIS script, where music4gpu's CMakeLists.txt looks
# for it (add_subdirectory).  On HPC sites that already provide a tuned Kokkos
# (Spack / module), skip this script and configure with -DKokkos_ROOT=<path>;
# the build falls back to find_package(Kokkos).
###############################################################################

KOKKOS_REPO="https://github.com/kokkos/kokkos.git"
# Clone next to this script regardless of the caller's cwd.
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/external/kokkos"

# Version: use $1 if given (reproducible / pinned), else the latest release tag.
# For CI / HPC reproducibility, PIN by passing a version: bash get_kokkos.sh 5.1.1
KOKKOS_TAG="$1"
if [ -z "$KOKKOS_TAG" ]; then
    echo "No version given - querying latest Kokkos release ..."
    KOKKOS_TAG=$(git ls-remote --tags --refs --sort=-v:refname "$KOKKOS_REPO" \
        | grep -oE 'refs/tags/[0-9]+\.[0-9]+\.[0-9]+$' | head -1 | sed 's|refs/tags/||')
    if [ -z "$KOKKOS_TAG" ]; then
        echo "ERROR: could not determine the latest release; pass one explicitly, e.g.:" >&2
        echo "       bash get_kokkos.sh 5.1.1" >&2
        exit 1
    fi
    echo "Latest release is ${KOKKOS_TAG}."
fi

if [ -d "$DEST" ]; then
    echo "Kokkos already present at ${DEST} - skipping (rm -rf to re-clone)."
    exit 0
fi

echo "Cloning Kokkos ${KOKKOS_TAG} -> ${DEST}"
git clone --depth=1 "$KOKKOS_REPO" --branch "$KOKKOS_TAG" "$DEST"

echo ""
echo "Kokkos ${KOKKOS_TAG} ready at ${DEST}"
echo "Build music4gpu with:  cmake -DUSE_KOKKOS=ON [other flags] .."
