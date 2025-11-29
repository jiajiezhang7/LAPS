import os, sys, glob, json
import h5py
root = "./data/preprocessed_gtea_m10/split1"
files = sorted(glob.glob(os.path.join(root, "*.hdf5")))
print(f"[CHECK] HDF5 count: {len(files)}")
errors = []
for p in files:
    try:
        with h5py.File(p, "r") as f:
            if "root" not in f:
                errors.append((p, "missing group: root")); continue
            g = f["root"]
            if "default" not in g:
                errors.append((p, "missing group: root/default")); continue
            d = g["default"]
            for ds in ("tracks", "vis"):
                if ds not in d:
                    errors.append((p, f"missing dataset: {ds}"))
                    continue
                x = d[ds]
                _shape = x.shape
                _dtype = str(x.dtype)
                if x.size == 0:
                    errors.append((p, f"empty dataset: {ds}"))
                _ = x[0:1]
    except Exception as e:
        errors.append((p, f"open/read failed: {e}"))

ok = (len(errors) == 0 and len(files) == 21)
print(json.dumps({"ok": ok, "n_files": len(files), "errors": errors}, indent=2))
if not ok:
    sys.exit(1)
