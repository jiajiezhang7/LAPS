import argparse, json
import h5py, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    args = ap.parse_args()
    p = args.h5

    out = {}
    with h5py.File(p, 'r') as f:
        out['root_keys'] = list(f.keys())
        grp = f['root/default']
        out['attrs'] = {k: grp.attrs.get(k) for k in grp.attrs.keys()}
        dset = grp['tracks']
        out['shape'] = tuple(dset.shape)
        T, H, N, _ = dset.shape
        vals = []
        for t0 in [0, max(0,T//3), max(0,2*T//3), max(0,T-2)]:
            win = dset[t0:t0+1]
            x = win[0]
            if x.shape[0] >= 2:
                diffs = np.diff(x, axis=0)
                mags = np.linalg.norm(diffs, axis=-1)
                median_per_step = np.median(mags, axis=1)
                vals.append(float(np.median(median_per_step)))
            else:
                vals.append(0.0)
        out['sample_median_vel_px_vals'] = vals
        out['sample_median_vel_px_mean'] = float(np.mean(vals)) if vals else 0.0
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

