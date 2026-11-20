#!/usr/bin/env bash
# Run pytest with the given arguments. On failure, write pytest's FAILED/ERROR
# summary lines as one GitHub Actions error annotation, so the failing tests
# can be read without a login (the annotations API is public for this
# repository, the job log is not).
set -o pipefail
log="$(mktemp)"
status=0
pytest -rfE "$@" | tee "$log" || status=$?
if [ "$status" -ne 0 ]; then
    lines="$(grep -E '^(FAILED|ERROR) ' "$log" | head -n 100)"
    summary="$(grep -E '^=+ .*(passed|failed|error).* =+$' "$log" | tail -n 1)"
    message="$(printf '%s\n%s\n' "$summary" "$lines" \
        | sed -e 's/%/%25/g' -e 's/\r/%0D/g' | awk '{printf "%s%%0A", $0}')"
    echo "::error title=pytest $*::${message}"
fi
rm -f "$log"
exit "$status"
