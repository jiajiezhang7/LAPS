import numpy as np
from torch.utils.data._utils.collate import default_collate
from amplify.loaders.custom_segments_dataset import CustomSegmentsDataset
root='/home/johnny/action_ws/data/preprocessed_gtea_m10/split1'
ds=CustomSegmentsDataset(root_dir=root, dataset_names=['custom_segments'], img_shape=(480,771), true_horizon=16, track_pred_horizon=16, keys_to_load=['tracks','images'])
batch=[ds[i] for i in range(2)]
for k in ['images','traj','vis']:
    xs=[{k: b[k]} for b in batch]
    print('try key:', k)
    try:
        out=default_collate(xs)
        v=out[k]
        print('  ok:', type(v), getattr(v,'shape',None), getattr(v,'dtype',None))
    except Exception as e:
        import traceback; traceback.print_exc()
print('done')

