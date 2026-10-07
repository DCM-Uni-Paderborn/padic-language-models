"""Frozen independent mechanism reproduction, executed on Spark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from padic_lm.continuous_tree import AffineDisk, direct_loss, norm_int


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execution", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    execution = json.loads(args.execution.read_text())
    for rel, pin in execution["source_sha256"].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == pin, rel
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    (args.output/"sources").mkdir()
    for rel in execution["source_sha256"]:
        (args.output/"sources"/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    (args.output/"execution.json").write_bytes(args.execution.read_bytes())
    start = time.monotonic()
    rng = np.random.default_rng(17)
    figure = AffineDisk(3, 1, radius=2/27, depth=6)
    figure.c[:] = 14*9
    x = np.array([[1]], dtype=np.int64); y = np.array([23], dtype=np.int64)
    trajectory = [dict(state=figure.state(), loss=direct_loss(figure, x, y))]
    for step in range(20):
        record = figure.update(x, "direct", rng, 8/81,
            lambda: direct_loss(figure, x, y), y=y)
        trajectory.append(dict(step=step, update=record, state=figure.state(), loss=direct_loss(figure, x, y)))
    write(args.output/"figure-trajectory.json", trajectory)
    fits = []
    for seed in (17, 29, 43):
        rng = np.random.default_rng(seed)
        theta = rng.integers(0, 3**5, size=3, dtype=np.int64)
        arrays = {"theta": theta}
        for part in ("train", "validation", "test"):
            xx = np.column_stack((rng.integers(0, 81, size=(512, 2), dtype=np.int64), np.ones(512, dtype=np.int64)))
            noise = rng.integers(0, 27, size=512, dtype=np.int64)
            arrays[part+"_x"] = xx; arrays[part+"_y"] = xx@theta+3**5*noise
        np.savez_compressed(args.output/f"regression-{seed}-data.npz", **arrays)
        m = AffineDisk(3, 3, radius=1, depth=5)
        trace = [dict(state=m.state(), loss=direct_loss(m, arrays["train_x"], arrays["train_y"]))]
        for step in range(100):
            if time.monotonic()-start > 300:
                raise TimeoutError("fixed300-second allocation")
            rec = m.update(arrays["train_x"], "direct", rng, 2.,
                lambda: direct_loss(m, arrays["train_x"], arrays["train_y"]), y=arrays["train_y"])
            trace.append(dict(step=step, update=rec, state=m.state(), loss=direct_loss(m, arrays["train_x"], arrays["train_y"])))
        write(args.output/f"regression-{seed}-trajectory.json", trace)
        evaluated = {}
        for part in ("train", "validation", "test"):
            err = m.denominator*norm_int(arrays[part+"_x"]@m.c-arrays[part+"_y"]*m.denominator, 3)
            evaluated[part] = dict(max_point_error=float(err.max()), direct_loss=direct_loss(m, arrays[part+"_x"], arrays[part+"_y"]))
        recovered = bool(np.all((m.c-theta*m.denominator) % (m.denominator*3**5) == 0))
        fits.append(dict(seed=seed, theta=theta.tolist(), state=m.state(), recovered_modulo_243=recovered,
                         accepted_updates=sum(t.get("update", {}).get("moved", False) for t in trace), metrics=evaluated))
        print(json.dumps(fits[-1]), flush=True)
    figure_pass = figure.c[0] == 23*9 and figure.r[0] <= 3**-6*(1+1e-12)
    gate = bool(figure_pass and all(f["recovered_modulo_243"] and f["accepted_updates"] > 0 and
                f["metrics"]["test"]["max_point_error"] <= 3**-5*(1+1e-12) for f in fits))
    write(args.output/"summary.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
          scope="independent mechanism reproduction, not full upstream experimental reproduction",
          figure_pass=bool(figure_pass), fits=fits, producer_gate=gate, elapsed_seconds=time.monotonic()-start))


if __name__ == "__main__":
    main()
