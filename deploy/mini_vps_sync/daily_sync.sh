#!/bin/sh
# Daily VPS -> mini market-data sync. Pulls new MNQ L2/depth + nq_l2_open from the
# VPS via the `vps` ssh alias (single hop). rsync is resumable and idempotent, so
# a partial current-day file is reconciled on the next run. Launched by launchd
# at 15:30 CT (30 min after the CME RTH close).
set -u
D="$HOME/vps_mirror"; LOG="$D/sync.log"
NTFY=$(grep -m1 NTFY_TOPIC "$HOME/wk_fwd/.env" 2>/dev/null | cut -d= -f2)
ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }
echo "=== sync start $(ts) ===" >> "$LOG"
n1=$(rsync -a --partial --stats -e "ssh -o BatchMode=yes" vps:/root/mnq_l2_data/ "$D/mnq_l2_data/" 2>>"$LOG" | grep "Number of files transferred" | grep -oE "[0-9]+" | head -1)
rc1=$?
n2=$(rsync -a --partial --stats -e "ssh -o BatchMode=yes" vps:/root/nq_l2_open/ "$D/nq_l2_open/" 2>>"$LOG" | grep "Number of files transferred" | grep -oE "[0-9]+" | head -1)
rc2=$?
tot=$(du -sh "$D" | cut -f1)
days=$(ls "$D"/mnq_l2_data/depth_MNQ_*.parquet 2>/dev/null | wc -l | tr -d " ")
msg="VPS sync: mnq +${n1:-0} nq +${n2:-0} files | ${days} depth days | ${tot} total | rc=$rc1/$rc2"
echo "$(ts)  $msg" >> "$LOG"
echo "=== sync done $(ts) ===" >> "$LOG"
if [ -n "$NTFY" ]; then
  prio="default"; [ "$rc1" != "0" ] || [ "$rc2" != "0" ] && prio="high"
  curl -s -m 10 -H "Title: VPS data sync" -H "Priority: $prio" -d "$msg" "https://ntfy.sh/$NTFY" >/dev/null 2>&1
fi
