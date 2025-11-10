from amplify.loaders.custom_segments_dataset import CustomSegmentsDataset

root='/home/johnny/action_ws/data/preprocessed_gtea_m10/split1'

ds = CustomSegmentsDataset(root_dir=root, dataset_names=['custom_segments'])
print('index_len=', len(ds.index_map))
first = ds.index_map[0]
print('first idx:', {k: first[k] for k in ['track_path','start_t','end_t','rollout_len'] if k in first})

sample = ds.load_tracks(first)
print('tracks shape:', sample['tracks'].shape)
print('vis present:', 'vis' in sample, 'vis shape:', sample['vis'].shape if 'vis' in sample else None)
print('OK')

