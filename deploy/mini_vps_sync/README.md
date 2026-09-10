# Mac mini → VPS daily market-data sync

Pulls new MNQ L2/depth + `nq_l2_open` from the research VPS to the always-on Mac
mini every day at **15:30 America/Chicago** (30 min after the CME RTH close).

## Where it runs
Mac mini `jg@100.113.240.72`. The VPS is reached via the `vps` ssh alias in the
mini's `~/.ssh/config` (single hop, `id_ed25519`). Data lands in `~/vps_mirror/`.

## Components (deployed on the mini, copies here for version control)
- `daily_sync.sh` → `~/vps_mirror/daily_sync.sh` — two idempotent `rsync -a
  --partial` pulls (mnq_l2_data, nq_l2_open), appends to `~/vps_mirror/sync.log`,
  and sends an ntfy push (topic from `~/wk_fwd/.env`; high priority on non-zero rc).
- `com.trading.vps-sync.plist` → `~/Library/LaunchAgents/` — launchd calendar
  job, `Hour 15 Minute 30`, `RunAtLoad false`, `/bin/sh -lc` (login shell so the
  homebrew PATH resolves — the one gotcha that makes hand-run plists do nothing
  on a timer).

## Install / manage (on the mini)
    launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.trading.vps-sync.plist
    launchctl kickstart gui/501/com.trading.vps-sync     # run now
    launchctl print    gui/501/com.trading.vps-sync      # inspect schedule
    launchctl bootout  gui/501/com.trading.vps-sync      # stop/unload

## Why launchd, not cron
macOS cron is deprecated and blocked by TCC from reading many paths. launchd is
the supported scheduler and auto-loads the agent at login/reboot. Verified the
scheduled path end-to-end via kickstart on 2026-09-09.

## Notes
- rsync is resumable and idempotent: the current day's still-growing file is
  reconciled on the next run, and a dropped link only re-fetches what's missing.
- Requires a logged-in `jg` GUI (aqua) session — the box already runs other
  LaunchAgents (amber-funnel), so that session exists.
