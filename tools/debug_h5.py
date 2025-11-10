import h5py, sys
p = sys.argv[1]
print("[DEBUG] open:", p)
with h5py.File(p, "r") as f:
    print("[DEBUG] keys:", list(f.keys()))
    if "root" in f:
        g = f["root"]
        print("[DEBUG] root keys:", list(g.keys()))
        if "default" in g:
            d = g["default"]
            print("[DEBUG] root/default keys:", list(d.keys()))
            for k in list(d.keys()):
                try:
                    obj = d[k]
                    print("  -", k, type(obj), getattr(obj, "shape", None), getattr(obj, "dtype", None))
                    # try small read
                    if hasattr(obj, "shape"):
                        _ = obj[0:1]
                        print("    [OK] slice read")
                except Exception as e:
                    print("    [ERR]", k, e)
