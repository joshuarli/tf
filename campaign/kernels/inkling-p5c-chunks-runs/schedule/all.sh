#!/bin/sh
set -eu
D=/Users/josh/repos/tf/.work/tmp/p5c/oracle
"$D/run-jobs.sh" "$D" "$D/jobs.txt"
for f in "$D"/main-served/*.bin; do
  n=$(basename "$f")
  case "$n" in harbour.bin|cells_code.bin) ;; *) cp "$f" "$D/lane-served/$n" ;; esac
done
"$D/run-jobs.sh" "$D" "$D/../probe-jobs.txt"
ls -la "$D/probe"
