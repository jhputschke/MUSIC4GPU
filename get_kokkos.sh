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
#   bash get_kokkos.sh            # pinned default (5.1.1) - reproducible
#   bash get_kokkos.sh 5.1.1      # an explicit release tag (same as default)
#   bash get_kokkos.sh latest     # track the newest release instead
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

# Version selection.  The DEFAULT is PINNED to the release this backend was
# developed and validated against, so stand-alone, CI and HPC builds are
# reproducible out of the box.  Override with an explicit tag, or pass "latest"
# to resolve the newest release at clone time.
KOKKOS_DEFAULT_TAG="5.1.1"
KOKKOS_TAG="$1"
if [ -z "$KOKKOS_TAG" ]; then
    KOKKOS_TAG="$KOKKOS_DEFAULT_TAG"
    echo "No version given - using pinned default ${KOKKOS_TAG}."
elif [ "$KOKKOS_TAG" = "latest" ]; then
    echo "Querying latest Kokkos release ..."
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
