#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INTERVAL_SECONDS="${BACKUP_INTERVAL_SECONDS:-1200}"
OUTPUT_DIR="${BACKUP_OUTPUT_DIR:-outputs/depatch_celeb_fbi}"
REMOTE="${BACKUP_REMOTE:-origin}"
BRANCH="${BACKUP_BRANCH:-training-backup}"
WORKTREE_DIR="${BACKUP_WORKTREE_DIR:-.git-training-backup-worktree}"

ARTIFACTS=(
  "$OUTPUT_DIR/latest_patch.pt"
  "$OUTPUT_DIR/latest_patch.png"
  "$OUTPUT_DIR/best_patch.pt"
  "$OUTPUT_DIR/best_patch.png"
  "$OUTPUT_DIR/history.csv"
  "$OUTPUT_DIR/training_progress.png"
  "$OUTPUT_DIR/training_progress.svg"
)

ensure_backup_worktree() {
  if [ -d "$WORKTREE_DIR/.git" ] || [ -f "$WORKTREE_DIR/.git" ]; then
    return 0
  fi

  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git worktree add "$WORKTREE_DIR" "$BRANCH"
    return 0
  fi

  if git ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null 2>&1; then
    git fetch "$REMOTE" "$BRANCH:$BRANCH"
    git worktree add "$WORKTREE_DIR" "$BRANCH"
    return 0
  fi

  git worktree add --detach "$WORKTREE_DIR" HEAD
  (
    cd "$WORKTREE_DIR"
    git switch --orphan "$BRANCH"
    git rm -rf . >/dev/null 2>&1 || true
    find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
    git commit --allow-empty -m "Initialize training backup branch"
    git push -u "$REMOTE" "$BRANCH"
  )
}

backup_once() {
  local existing=()
  local path

  ensure_backup_worktree

  for path in "${ARTIFACTS[@]}"; do
    if [ -e "$path" ]; then
      existing+=("$path")
    fi
  done

  if [ "${#existing[@]}" -eq 0 ]; then
    echo "No backup artifacts found in $OUTPUT_DIR"
    return 0
  fi

  for path in "${existing[@]}"; do
    mkdir -p "$WORKTREE_DIR/$(dirname "$path")"
    cp -p "$path" "$WORKTREE_DIR/$path"
  done

  (
    cd "$WORKTREE_DIR"
    git add -A

    if git diff --cached --quiet; then
      echo "No artifact changes to commit"
      return 0
    fi

    git commit -m "Backup training artifacts $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    git push "$REMOTE" "$BRANCH"
  )
}

if [ "${1:-}" = "--once" ]; then
  backup_once
  exit 0
fi

while true; do
  if ! backup_once; then
    echo "Backup attempt failed; retrying in ${INTERVAL_SECONDS}s" >&2
  fi
  sleep "$INTERVAL_SECONDS"
done
