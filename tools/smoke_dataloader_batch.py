import traceback
import numpy as np
from torch.utils.data import DataLoader
from amplify.loaders.custom_segments_dataset import CustomSegmentsDataset

root='/home/johnny/action_ws/data/preprocessed_gtea_m10/split1'

ds = CustomSegmentsDataset(root_dir=root, dataset_names=['custom_segments'], img_shape=(480,771), true_horizon=16, track_pred_horizon=16, keys_to_load=['tracks','images'])
print('index_len=', len(ds.index_map))

def collate_debug(batch):
    # Inspect dtypes
    keys = batch[0].keys()
    print('batch keys:', keys)
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        types = [type(v) for v in vals[:3]]
        print('key', k, 'types', types)
        if isinstance(vals[0], np.ndarray):
            print('  dtype:', vals[0].dtype, 'shape:', vals[0].shape)
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)

loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_debug)

for i, b in enumerate(loader):
    print('got batch', i)
    for k, v in b.items():
        try:
            print('  ', k, type(v), getattr(v, 'shape', None))
        except Exception:
            pass
    break
print('OK')

