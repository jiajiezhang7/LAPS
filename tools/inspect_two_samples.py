import numpy as np
from amplify.loaders.custom_segments_dataset import CustomSegmentsDataset
root='./data/preprocessed_gtea_m10/split1'
ds=CustomSegmentsDataset(root_dir=root, dataset_names=['custom_segments'], img_shape=(480,771), true_horizon=16, track_pred_horizon=16, keys_to_load=['tracks','images'])
for i in range(2):
    s=ds[i]
    print(f'keys{i}:', s.keys())
    for k,v in s.items():
        print(f' {i} {k}:', type(v), getattr(v,'dtype',None), getattr(v,'shape',None))
print('done')

