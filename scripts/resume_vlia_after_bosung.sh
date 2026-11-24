#!/usr/bin/env bash
# Trang 21/11: when every bosung block is done, VLIA 100k training continues (from its newest complete checkpoint).
# If bosung stops with FAILED, nothing is resumed (waits for Trang).
L=/media/dhqg/d3/logs/bosung_2026-11-20.log
O=/media/dhqg/d3/vlia_outputs/train_100k/vlia
STATE=/media/dhqg/d3/vlia_outputs/ke_hoach_cuoi/STATE.md
# only lines after the last REAL start count (a dry run also prints "ALL BLOCKS DONE")
since() { awk '/start \(dry-run 0/ {buf=""} {buf=buf $0 "\n"} END {printf "%s", buf}' $L 2>/dev/null | grep -v "dry-run 1"; }
while ! since | grep -q "ALL BLOCKS DONE"; do
  since | grep -q "FAILED" && { echo "$(date '+%F %T') bosung FAILED: VLIA not resumed"; exit 1; }
  sleep 300
done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
rm -f $O/PAUSE $O/STOPPED_BY_TRANG
echo "$(date '+%F %T') | Part 4 | train100k vlia | resumed after bosung (Trang 21/11) | $O | -" >> $STATE
tmux new -d -s vlia100k "bash /media/dhqg/d3/vlia-100k/scripts/run_train100k_chain.sh vlia"
echo "$(date '+%F %T') VLIA chain started"
