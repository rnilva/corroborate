#!/bin/bash
# Generate partner sweep configs (seed_offset=15) for the 6 MLP-env canonical
# full-Q sweeps. Resumes the seeds 0-14 sweeps with seeds 15-29 so the
# combined corpus has the canonical n=30 per arm.
set -euo pipefail

cd /workspace/corroborate/experiments/configs

for base in acrobot cartpole fr metamaze mountaincar lunarlander; do
    src="${base}_g099_canonical_n_eps20_ckpt.yaml"
    dst="${base}_g099_canonical_n_eps20_ckpt_seeds15to29.yaml"
    if [ ! -f "$src" ]; then
        echo "MISSING: $src" >&2
        continue
    fi
    sed -e "s|name: ${base}_g099_canonical_n_eps20_ckpt$|name: ${base}_g099_canonical_n_eps20_ckpt_seeds15to29|" \
        -e "s|out_dir: experiments/data/${base}_g099_canonical_n_eps20_ckpt$|out_dir: experiments/data/${base}_g099_canonical_n_eps20_ckpt_seeds15to29|" \
        -e "s|archive_remote: s3://corroborate-archive/${base}_g099_canonical_n_eps20_ckpt$|archive_remote: s3://corroborate-archive/${base}_g099_canonical_n_eps20_ckpt_seeds15to29|" \
        -e 's|n_seeds: 15, chunk_size: 15|n_seeds: 15, chunk_size: 15, seed_offset: 15|' \
        "$src" > "$dst"
    echo "wrote $dst"
done
