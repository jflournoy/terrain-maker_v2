#!/usr/bin/env bash
# Install a git post-merge hook (runs after `git pull`) for scripts/verify-local.sh.
#
# By default the hook only prints a reminder. To run verification automatically after each
# pull, set TERRAIN_VERIFY_ON_PULL in your shell profile:
#   export TERRAIN_VERIFY_ON_PULL=quick   # deps + tests
#   export TERRAIN_VERIFY_ON_PULL=full    # + mock and real-data pins
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
HOOK=.git/hooks/post-merge
if [ -e "$HOOK" ] && ! grep -q "verify-local.sh" "$HOOK"; then
  echo "$HOOK already exists and is not ours; not overwriting." >&2
  exit 1
fi
cat > "$HOOK" <<'HOOK'
#!/usr/bin/env bash
# Installed by scripts/install-verify-hook.sh
case "${TERRAIN_VERIFY_ON_PULL:-}" in
  quick) scripts/verify-local.sh --quick ;;
  full) scripts/verify-local.sh ;;
  *) echo "terrain-maker updated. Verify with: scripts/verify-local.sh  (add --render for a visual check)" ;;
esac
HOOK
chmod +x "$HOOK"
echo "Installed $HOOK"
