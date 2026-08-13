#!/usr/bin/env bash

set -euo pipefail

SOURCE_ROOT="/home/d013/桌面/SonicStar"
TARGET_REPO_URL="${TARGET_REPO_URL:-git@github.com:Ju6276/SonicStar-DEV.git}"
TARGET_BRANCH=""
TARGET_BASE_BRANCH="${TARGET_BASE_BRANCH:-}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Sync from SonicStar excluding playground}"
KEEP_WORKTREE="${KEEP_WORKTREE:-0}"
GIT_NAME="${GIT_NAME:-}"
GIT_EMAIL="${GIT_EMAIL:-}"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") --branch <target-branch> [options]

Sync this SonicStar workspace into Ju6276/SonicStar-DEV while excluding:
  starVLA/starVLA/playground/

This script never pushes to main directly. It only creates or updates the
branch passed via --branch.

Options:
  --branch <name>         Target branch to create/update in SonicStar-DEV. Required.
  --base-branch <name>    Base branch in SonicStar-DEV. Defaults to remote default branch.
  --repo-url <url>        Target repo URL.
  --message <msg>         Commit message.
  --git-name <name>       Git author name to use for the sync commit.
  --git-email <email>     Git author email to use for the sync commit.
  --keep-worktree         Do not delete the temporary cloned repo after completion.
  --help                  Show this help.

Environment overrides:
  TARGET_REPO_URL
  TARGET_BASE_BRANCH
  COMMIT_MESSAGE
  GIT_NAME
  GIT_EMAIL
  KEEP_WORKTREE=1

Examples:
  $(basename "$0") --branch sonicstar-sync-june25
  $(basename "$0") --branch sonicstar-sync-june25 --base-branch main --message "Sync June 25"
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --branch)
        TARGET_BRANCH="${2:-}"
        shift 2
        ;;
      --base-branch)
        TARGET_BASE_BRANCH="${2:-}"
        shift 2
        ;;
      --repo-url)
        TARGET_REPO_URL="${2:-}"
        shift 2
        ;;
      --message)
        COMMIT_MESSAGE="${2:-}"
        shift 2
        ;;
      --git-name)
        GIT_NAME="${2:-}"
        shift 2
        ;;
      --git-email)
        GIT_EMAIL="${2:-}"
        shift 2
        ;;
      --keep-worktree)
        KEEP_WORKTREE="1"
        shift
        ;;
      --help|-h)
        print_usage
        exit 0
        ;;
      *)
        printf 'Unknown argument: %s\n\n' "$1" >&2
        print_usage >&2
        exit 1
        ;;
    esac
  done

  if [[ -z "$TARGET_BRANCH" ]]; then
    printf 'Missing required argument: --branch\n\n' >&2
    print_usage >&2
    exit 1
  fi
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$cmd" >&2
    exit 1
  fi
}

detect_default_branch() {
  git ls-remote --symref "$TARGET_REPO_URL" HEAD \
    | awk '/^ref:/ {sub("refs/heads/", "", $2); print $2; exit}'
}

resolve_git_identity() {
  if [[ -z "$GIT_NAME" ]]; then
    GIT_NAME="$(git config --global --get user.name || true)"
  fi
  if [[ -z "$GIT_EMAIL" ]]; then
    GIT_EMAIL="$(git config --global --get user.email || true)"
  fi

  if [[ -z "$GIT_NAME" || -z "$GIT_EMAIL" ]]; then
    printf 'Missing git identity for commit.\n' >&2
    printf 'Pass --git-name and --git-email, or set global git config.\n' >&2
    exit 1
  fi
}

main() {
  parse_args "$@"

  require_command git
  require_command rsync
  require_command mktemp
  resolve_git_identity

  local base_branch
  if [[ -n "$TARGET_BASE_BRANCH" ]]; then
    base_branch="$TARGET_BASE_BRANCH"
  else
    base_branch="$(detect_default_branch)"
  fi

  if [[ -z "$base_branch" ]]; then
    printf 'Failed to detect default branch for %s\n' "$TARGET_REPO_URL" >&2
    exit 1
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/sonicstar-dev-sync.XXXXXX)"
  local cleanup_tmp="1"
  if [[ "$KEEP_WORKTREE" == "1" ]]; then
    cleanup_tmp="0"
  fi

  cleanup() {
    if [[ "$cleanup_tmp" == "1" ]] && [[ -d "$tmp_dir" ]]; then
      rm -rf "$tmp_dir"
    fi
  }
  trap cleanup EXIT

  printf '[1/6] Cloning target repo: %s\n' "$TARGET_REPO_URL"
  GIT_LFS_SKIP_SMUDGE=1 git clone "$TARGET_REPO_URL" "$tmp_dir/repo"

  cd "$tmp_dir/repo"
  git config user.name "$GIT_NAME"
  git config user.email "$GIT_EMAIL"

  printf '[2/6] Preparing target branch: %s (base: %s)\n' "$TARGET_BRANCH" "$base_branch"
  git fetch origin
  if git ls-remote --exit-code --heads origin "$TARGET_BRANCH" >/dev/null 2>&1; then
    git checkout -B "$TARGET_BRANCH" "origin/$TARGET_BRANCH"
  else
    git checkout -B "$TARGET_BRANCH" "origin/$base_branch"
  fi

  printf '[3/6] Cleaning target worktree contents...\n'
  find . -mindepth 1 -maxdepth 1 \
    ! -name .git \
    -exec rm -rf {} +

  printf '[4/6] Syncing files from source workspace...\n'
  rsync -a \
    --delete \
    --exclude '.git/' \
    --exclude '.agents/' \
    --exclude '.codex/' \
    --exclude 'starVLA/starVLA/playground/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    --exclude '*.egg-info/' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.DS_Store' \
    "${SOURCE_ROOT}/" .

  printf '[5/6] Creating commit if there are changes...\n'
  git add -A
  if git diff --cached --quiet; then
    printf 'No changes to commit after sync.\n'
  else
    git commit -m "$COMMIT_MESSAGE"
    printf '[6/6] Pushing branch: %s\n' "$TARGET_BRANCH"
    git push -u origin "$TARGET_BRANCH"
  fi

  printf '\nDone.\n'
  printf 'Target repo: %s\n' "$TARGET_REPO_URL"
  printf 'Target branch: %s\n' "$TARGET_BRANCH"
  printf 'Main branch untouched. Only branch "%s" was updated.\n' "$TARGET_BRANCH"
  if [[ "$cleanup_tmp" == "0" ]]; then
    printf 'Temporary worktree kept at: %s/repo\n' "$tmp_dir"
  fi
}

main "$@"
