"""Record runs and select measured throughput winners, scoped by STUDY_ID."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import time


# (environments per GPU, GLOBAL PPO minibatch). All keep six SAPG blocks
# and divide both the critic and augmented actor rollout without remainders.
TRIALS = [(6144, 16384), (6144, 32768),
          (12288, 16384), (12288, 32768), (12288, 65536),
          (24576, 16384), (24576, 32768), (24576, 65536)]
MIN_GLOBAL_MINIBATCH = 32768
REQUIRED_TRIALS = [pair for pair in TRIALS if pair[1] >= MIN_GLOBAL_MINIBATCH]
# Optional staged probes; only retained baseline trials are required for best.
# Completed probes participate in winner selection just like baseline trials.
EXTENDED_TRIALS = [(24576, 131072), (49152, 131072), (49152, 262144)]
ALL_TRIALS = TRIALS + EXTENDED_TRIALS


def steady_fps(text, warmup=20):
    values = [float(v.replace(',', '')) for v in
              re.findall(r'fps total\s*:\s*([\d,.]+)', text)][warmup:]
    if len(values) < 20 or any(v <= 0 for v in values):
        return None
    # Each epoch has the same transition count within a trial. Harmonic mean
    # is total transitions / total elapsed time, unlike an arithmetic mean.
    return dict(fps=statistics.harmonic_mean(values), epochs=len(values),
                fps_median=statistics.median(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    trial = sub.add_parser('trial')
    trial.add_argument('index', type=int, choices=range(len(ALL_TRIALS)))
    record = sub.add_parser('record')
    record.add_argument('--status', choices=['running', 'completed', 'failed'], required=True)
    for command in ('best', 'report'):
        q = sub.add_parser(command)
        q.add_argument('--root', type=Path, default=Path('debug_outputs/scaling_laws'))
        q.add_argument('--study', default='scaling_laws_shared_privileged_v1')
        q.add_argument('--model', required=command == 'best')
    args = p.parse_args()
    if args.command == 'trial':
        print(*ALL_TRIALS[args.index])
        return
    if args.command == 'record':
        root = Path(os.environ['SCALING_RUN_DIR'])
        path = root / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        result.update(status=args.status, phase=os.environ['PHASE'], study=os.environ['STUDY_ID'],
                      model=f"d{os.environ['D_MODEL']}_l{os.environ['TRANSFORMER_LAYERS']}",
                      envs_per_gpu=int(os.environ['NUM_ENVS_PER_GPU']),
                      global_minibatch=int(os.environ['GLOBAL_MINIBATCH']),
                      seed=int(os.environ['SEED']), world_size=2,
                      actor_critic='shared', privileged_policy=True, value_head_units=[512, 256])
        result.setdefault('started_at', time.time())
        if args.status != 'running':
            result['finished_at'] = time.time()
            logs = sorted(root.glob('torchrun/**/0/stdout.log'))
            if len(logs) == 1:
                result['steady'] = steady_fps(logs[0].read_text())
        path.write_text(json.dumps(result, indent=2) + '\n')
        return
    rows = []
    for path in args.root.glob('*/result.json'):
        r = json.loads(path.read_text())
        if r['study'] == args.study and r['phase'] == 'tune' and (not args.model or r['model'] == args.model):
            r['path'] = str(path)
            rows.append(r)
    if args.command == 'report':
        for r in sorted(rows, key=lambda r: (r['model'], r['envs_per_gpu'], r['global_minibatch'])):
            print(json.dumps(r))
    else:
        eligible = [r for r in rows if r['status'] == 'completed' and r.get('steady')
                    and r['global_minibatch'] >= MIN_GLOBAL_MINIBATCH]
        measured = {(r['envs_per_gpu'], r['global_minibatch']) for r in eligible}
        failed = {(r['envs_per_gpu'], r['global_minibatch']) for r in rows if r['status'] == 'failed'}
        missing = set(REQUIRED_TRIALS) - measured - failed
        if missing:
            p.error(f'Unfinished/unmeasured trials: {sorted(missing)}. Inspect logs before selecting.')
        if not eligible:
            p.error('No completed trials with at least 20 steady-state epochs.')
        # Median across repeats prevents selecting an unusually fast outlier.
        groups = {}
        for r in eligible:
            groups.setdefault((r['envs_per_gpu'], r['global_minibatch']), []).append(r['steady']['fps'])
        winner = max(groups, key=lambda pair: statistics.median(groups[pair]))
        print(*winner)


if __name__ == '__main__':
    main()
