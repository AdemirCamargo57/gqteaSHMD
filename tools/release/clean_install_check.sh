#!/usr/bin/env bash
# Clean-install check of a release (release spec 2026-11-25, section 5.1; plan PR1, PR3).
#
# Usage, in WSL with gqteashmd-dev active, from the repository:
#   tools/release/clean_install_check.sh VERSION [LOCK_FILE]
# Builds the wheel from a clean clone of HEAD, installs it (not editable) into a fresh
# environment created from the lock file, runs tools/release/water_check.toml through the
# analysis and checks the version line and provenance. Everything goes under
# ${RELEASE_CHECK_ROOT:-$HOME/release_check}/VERSION. Exit 0 and PASS when every check passes.
set -euo pipefail

version="${1:?usage: clean_install_check.sh VERSION [LOCK_FILE]}"
repo="$(git rev-parse --show-toplevel)"
lock="${2:-$repo/environment.lock.txt}"
base="${RELEASE_CHECK_ROOT:-$HOME/release_check}/$version"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

# `conda` is a shell function in an interactive shell; a script uses the executable it names
conda="${CONDA_EXE:-conda}"
command -v "$conda" >/dev/null || fail "conda not found: activate gqteashmd-dev"

# 1. the version at HEAD and a clean tree
head="$(git -C "$repo" rev-parse HEAD)"
source_version="$(git -C "$repo" show HEAD:src/gqteashmd/__init__.py \
    | sed -n 's/^__version__ = "\(.*\)"$/\1/p')"
[ "$source_version" = "$version" ] || fail "VERSION $version differs from __version__ $source_version at HEAD"
[ -z "$(git -C "$repo" status --porcelain)" ] || fail "the working tree of $repo is not clean"
[ -f "$lock" ] || fail "no lock file $lock"
lock="$(realpath "$lock")"
python -c "import hatchling" 2>/dev/null || fail "hatchling is missing: activate gqteashmd-dev"
[ ! -e "$base" ] || fail "$base exists; move it aside first"

# 2. a clean clone of HEAD and the wheel
mkdir -p "$base"
echo "clone $head into $base/src"
git clone --quiet "$repo" "$base/src"
git -C "$base/src" checkout --quiet "$head"
(cd "$base/src" && pip wheel . --no-deps --no-build-isolation --quiet -w "$base/dist")
wheel="$(ls "$base"/dist/gqteashmd-"$version"-*.whl)"
echo "built $wheel"

# 3. a fresh environment from the lock file, and the wheel installed into it
echo "create $base/env from $lock"
"$conda" create --yes --quiet -p "$base/env" --file "$lock" >/dev/null
"$conda" run -p "$base/env" pip install --no-deps --quiet "$wheel"
"$conda" list -p "$base/env" --json > "$base/conda_list.json"
# the tested environment, which the lock file was exported from: pip-installed runtime
# dependencies there would be missing from the lock
"$conda" list -n gqteashmd-dev --json > "$base/dev_conda_list.json"

# 4 and 5. the version line and a run, from outside the repository
mkdir -p "$base/project"
cp "$repo/tools/release/water_check.toml" "$repo/tools/release/water.xyz" "$base/project/"
cd "$base/project"
version_line="$("$conda" run -p "$base/env" gqteashmd --version)" || version_line=""
echo "version line: $version_line"
echo "run water_check.toml (log: $base/project/run.log)"
set +e
"$conda" run -p "$base/env" gqteashmd run water_check.toml > run.log 2>&1
run_exit=$?
set -e

# 6. every check into report.json, then PASS or FAIL
python "$repo/tools/release/release_checks.py" \
    --version "$version" --commit "$head" --version-line "$version_line" \
    --conda-list "$base/conda_list.json" --dev-conda-list "$base/dev_conda_list.json" \
    --provenance "$base/project/run/.gqteashmd/provenance.json" \
    --analysis "$base/project/run/07_analysis/analysis.json" \
    --run-exit "$run_exit" --out "$base/report.json"
