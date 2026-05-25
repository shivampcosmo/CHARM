#!/usr/bin/env python
"""
Run CHARM inference over the held-out test simulations.

Example:
    python charm/inferers/run_test_inference_v2.py \
        --config run_configs/TRAIN_CHARM_JOINT_v2.yaml \
        --sim_start 1900 \
        --sim_end 2000 \
        --num_workers 8 \
        --device cpu

`--sim_end` is exclusive, so the default test range 1900..1999 contains
100 simulations.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import subprocess
import sys
import time
from dataclasses import dataclass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'charm'))

from config_loader import load_config


@dataclass(frozen=True)
class Job:
    sim_id: int
    command: list[str]
    log_path: str
    out_path: str
    env: dict[str, str]


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run run_inference_v2.py over the held-out test sims.',
    )
    p.add_argument('--config', required=True,
                   help='Path to YAML config.')
    p.add_argument('--sim_start', type=int, default=None,
                   help='First simulation id. Defaults to nsims_train + nsims_val.')
    p.add_argument('--sim_end', type=int, default=None,
                   help='Exclusive upper simulation id. Defaults to sim_start + nsims_test.')
    p.add_argument('--sim_ids', nargs='*', type=int, default=None,
                   help='Explicit simulation ids. Overrides --sim_start/--sim_end.')
    p.add_argument('--num_workers', type=int, default=1,
                   help='Number of simulations to run concurrently.')
    p.add_argument('--device', default='cpu',
                   help='Device passed to run_inference_v2.py. Use cpu for CPU workers.')
    p.add_argument('--threads_per_worker', type=int, default=1,
                   help='CPU BLAS/OpenMP/Torch threads exposed to each worker.')
    p.add_argument('--checkpoint', default=None,
                   help='Optional checkpoint path passed through to run_inference_v2.py.')
    p.add_argument('--output_dir', default=None,
                   help='Output directory for mock catalogs. Defaults to '
                        '<checkpoint_dir>/inference/.')
    p.add_argument('--binary_target_prior', type=float, default=None,
                   help='Optional value passed through to run_inference_v2.py.')
    p.add_argument('--binary_prior_calibrator', default=None,
                   help='Optional calibrator path passed through to run_inference_v2.py.')
    p.add_argument('--binary_train_prior', type=float, default=None,
                   help='Optional trained-prior override passed through to run_inference_v2.py.')
    p.add_argument('--overwrite', action='store_true',
                   help='Rerun simulations even if their mock catalog already exists.')
    return p.parse_args()


def _default_sim_ids(cfg: dict, sim_start: int | None, sim_end: int | None) -> list[int]:
    sc = cfg['sim_settings']
    if sim_start is None:
        sim_start = int(sc.get('nsims_train', 1800)) + int(sc.get('nsims_val', 100))
    if sim_end is None:
        sim_end = sim_start + int(sc.get('nsims_test', 100))
    if sim_end <= sim_start:
        raise ValueError(f'Empty sim range: [{sim_start}, {sim_end})')
    return list(range(sim_start, sim_end))


def _resolve_output_dir(cfg: dict, output_dir: str | None) -> str:
    if output_dir:
        return os.path.abspath(output_dir)
    ckpt_dir = os.path.join(_REPO_ROOT, cfg['train_settings']['checkpoint_dir'])
    return os.path.abspath(os.path.join(ckpt_dir, 'inference'))


def _build_jobs(args, cfg: dict, sim_ids: list[int], output_dir: str) -> tuple[list[Job], list[int]]:
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    threads = str(max(1, int(args.threads_per_worker)))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'TORCH_NUM_THREADS'):
        env[key] = threads

    jobs = []
    skipped = []
    script = os.path.join(_REPO_ROOT, 'charm', 'inferers', 'run_inference_v2.py')

    for sim_id in sim_ids:
        out_path = os.path.join(output_dir, f'mock_catalog_sim{sim_id:04d}.npz')
        if os.path.exists(out_path) and not args.overwrite:
            skipped.append(sim_id)
            continue

        cmd = [
            sys.executable, script,
            '--config', os.path.abspath(args.config),
            '--sim_id', str(sim_id),
            '--device', args.device,
            '--output_dir', output_dir,
        ]
        if args.checkpoint:
            cmd += ['--checkpoint', os.path.abspath(args.checkpoint)]
        if args.binary_target_prior is not None:
            cmd += ['--binary_target_prior', str(args.binary_target_prior)]
        if args.binary_prior_calibrator:
            cmd += ['--binary_prior_calibrator',
                    os.path.abspath(args.binary_prior_calibrator)]
        if args.binary_train_prior is not None:
            cmd += ['--binary_train_prior', str(args.binary_train_prior)]

        log_path = os.path.join(log_dir, f'inference_sim{sim_id:04d}.log')
        jobs.append(Job(sim_id, cmd, log_path, out_path, env))

    return jobs, skipped


def _run_job(job: Job) -> tuple[int, int, float, str]:
    t0 = time.time()
    with open(job.log_path, 'w') as log:
        log.write(' '.join(job.command) + '\n\n')
        log.flush()
        proc = subprocess.run(
            job.command,
            cwd=_REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=job.env,
            check=False,
        )
    return job.sim_id, proc.returncode, time.time() - t0, job.log_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    sim_ids = args.sim_ids if args.sim_ids else _default_sim_ids(
        cfg, args.sim_start, args.sim_end)
    output_dir = _resolve_output_dir(cfg, args.output_dir)
    jobs, skipped = _build_jobs(args, cfg, sim_ids, output_dir)

    print(f'Output directory: {output_dir}', flush=True)
    print(f'Requested simulations: {len(sim_ids)}', flush=True)
    if skipped:
        print(f'Skipping {len(skipped)} existing catalogs '
              f'({min(skipped):04d}..{max(skipped):04d}). '
              f'Use --overwrite to rerun.', flush=True)
    if not jobs:
        print('Nothing to run.', flush=True)
        return

    n_workers = max(1, min(int(args.num_workers), len(jobs)))
    print(f'Running {len(jobs)} inference jobs with {n_workers} worker(s) '
          f'on device={args.device}. Logs: {os.path.join(output_dir, "logs")}',
          flush=True)

    failures = []
    with futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut_to_sim = {ex.submit(_run_job, job): job.sim_id for job in jobs}
        for fut in futures.as_completed(fut_to_sim):
            sim_id, returncode, elapsed, log_path = fut.result()
            if returncode == 0:
                print(f'[{sim_id:04d}] done in {elapsed/60:.1f} min', flush=True)
            else:
                failures.append((sim_id, returncode, log_path))
                print(f'[{sim_id:04d}] FAILED with return code {returncode}; '
                      f'see {log_path}', flush=True)

    if failures:
        failed = ', '.join(f'{sim:04d}' for sim, _, _ in failures)
        raise SystemExit(f'{len(failures)} inference job(s) failed: {failed}')

    print('All requested inference jobs completed.', flush=True)


if __name__ == '__main__':
    main()
