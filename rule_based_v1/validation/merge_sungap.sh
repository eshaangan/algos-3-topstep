#!/bin/bash
# Merge staged Sunday-gap files into each instrument's raw set.
# Non-destructive: cp -n, plus an explicit collision report first.
set -u
D=~/.svc-3hKye0
for r in hist_1m24_zn hist_1m24_mgc hist_1m24_mcl hist_1m24_m6e hist_1m24_mes; do
  S=$D/${r}_sungap
  [ -d "$S" ] || { echo "$r: no staging dir"; continue; }
  n=$(ls "$S" 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -eq 0 ] && { echo "$r: nothing staged"; continue; }
  c=0
  for f in "$S"/*.parquet; do
    b=$(basename "$f"); [ -e "$D/$r/$b" ] && c=$((c+1))
  done
  before=$(ls "$D/$r" | wc -l | tr -d ' ')
  cp -n "$S"/*.parquet "$D/$r"/
  after=$(ls "$D/$r" | wc -l | tr -d ' ')
  echo "$r: staged=$n collisions=$c  files $before -> $after"
done
