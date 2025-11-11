import argparse, json
import h5py

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    args = ap.parse_args()
    p = args.h5
    out = {}
    with h5py.File(p, 'r') as f:
        out['keys'] = list(f.keys())
        grp = f['root/default']
        out['attr_keys'] = list(grp.attrs.keys())
        out['height'] = grp.attrs.get('height')
        out['width'] = grp.attrs.get('width')
        out['has_tracks'] = 'tracks' in grp
        out['has_vis'] = 'vis' in grp
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

