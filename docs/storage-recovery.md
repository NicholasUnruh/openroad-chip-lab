# Compiled build cache storage and recovery

On October 9, 2026 UTC, **290** inactive `.gch`, `.o` and `.a` files from `build/`
were archived to Drive and removed locally after complete recovery verification,
reclaiming **1,590,665,216 allocated bytes**.

RTL, scripts, generated source, GDS layouts, waveforms, logs, metrics and JSON
results remain local. The [recorded V4 evidence](training-v4-results.md) is unchanged.
Normal build commands can regenerate the removed compiled caches.

[Completion receipt](../../../IBKR/ibkr-cnn-rl-research/research/provenance/2026-10-09-project-storage-export-completion.json)
pins exact manifests, remote checksums and recovery evidence.
[Drive archive](https://drive.google.com/drive/folders/1YAOSUqS50c0vpQX8jrUDO_wmOIKU2fx4).

## Recover exact previous cache files

Allow space for the archive and extraction plus the active workload's disk reserve.

```bash
restore_stage=$(mktemp -d /home/nunruh/openroad-restore.XXXXXX)
export_remote='gdrive:ibkr-cnn-rl-research/storage-export/20261009T051917Z'
rclone copy "$export_remote/openroad-compiled-cache" "$restore_stage" --immutable --transfers 1
printf '%s\n' "$restore_stage"
```

Follow `RESTORE.txt`: verify payload/manifest SHA-256 against `COMPLETED.json`,
extract into staging with Python `tarfile` and `filter="data"`, then check every
member's path, size and hash. Restore verified files to the recorded `build/` root
while inactive, preserving newer and retained files.
