import h5py
import numpy as np
from tqdm import tqdm
from amplify.loaders.real_dataset import TASKS

def get_action_stats(data_path):
    """
    hdf5 file format:
    data/
        demo_0/
            {action_keys}
            obs/
                {obs_keys}
        demo_1/
        ...
    text_emb
    """
    with h5py.File(data_path, "r") as f:
        action_data_lists = {}
        for demo_key in tqdm(f["data"]):
            for key in f["data"][demo_key]:
                if "action" not in key:
                    continue
                action_data = f["data"][demo_key][key][()]
                if key not in action_data_lists:
                    action_data_lists[key] = []
                action_data_lists[key].append(action_data)
    for key in action_data_lists:
        action_data_lists[key] = np.concatenate(action_data_lists[key], axis=0)
    action_stats = {
        action_key: {
            "mean": np.mean(action_data_lists[action_key], axis=0),
            "std": np.std(action_data_lists[action_key], axis=0),
            # 99th percentile min/max
            "min": np.percentile(action_data_lists[action_key], 1, axis=0),
            "max": np.percentile(action_data_lists[action_key], 99, axis=0),
        } for action_key in action_data_lists
    }
                
    return action_stats

if __name__ == "__main__":
    for task in TASKS:
        task = task.replace(" ", "_")
        data_path = f"preprocessed_data/real_robot_processed/{task}.hdf5"
        action_stats = get_action_stats(data_path)
        # print(action_stats)
        
        # Save to file
        output_path = f"preprocessed_data/real_robot_processed/{task}_action_stats.npy"
        np.save(output_path, action_stats)
        print(f"Saved action stats to {output_path}")
    