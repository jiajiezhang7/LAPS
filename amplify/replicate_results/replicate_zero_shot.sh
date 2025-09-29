# Motion Tokenizer
python train_motion_tokenizer.py run_name=libero_all_motion_tokenizer train_datasets=[libero_10:traj1.0,libero_90:traj1.0,libero_object:traj1.0,libero_spatial:traj1.0,libero_goal:traj1.0] val_datasets=[libero_10:traj-0.1]

# Forward Dynamics (uses Motion Tokenizer checkpoint)
python train_forward_dynamics.py run_name=libero_all_forward_dynamics forward_dynamics.motion_tokenizer.checkpoint=checkpoints/motion_tokenizer/libero_all_motion_tokenizer/latest.pt train_datasets=[libero_10:traj1.0,libero_90:traj1.0,libero_object:traj1.0,libero_spatial:traj1.0,libero_goal:traj1.0] val_datasets=[libero_10:traj-0.1]

# Inverse Dynamics (uses Motion Tokenizer and Forward Dynamics checkpoint)
python train_inverse_dynamics.py run_name=libero_90_inverse_dynamics motion_tokenizer_checkpoint=checkpoints/motion_tokenizer/libero_all_motion_tokenizer/latest.pt forward_dynamics_checkpoint=checkpoints/forward_dynamics/libero_all_forward_dynamics/latest.pt train_datasets=[libero_90:action1.0] val_datasets=[libero_90:action-0.1]

# Bundle
python -m amplify.bundle_amplify --mt_ckpt checkpoints/motion_tokenizer/libero_all_motion_tokenizer/latest.pt --fd_ckpt checkpoints/forward_dynamics/libero_all_forward_dynamics/latest.pt --id_ckpt checkpoints/inverse_dynamics/libero_90_inverse_dynamics_seed_0/latest.pt --name libero_zero_shot

# Eval
python eval_libero.py dataset=[libero_10] run_name=libero_zero_shot_libero_10 amplify_checkpoint=checkpoints/AMPLIFY/libero_zero_shot.pt
python eval_libero.py dataset=[libero_object] run_name=libero_zero_shot_libero_object amplify_checkpoint=checkpoints/AMPLIFY/libero_zero_shot.pt
python eval_libero.py dataset=[libero_spatial] run_name=libero_zero_shot_libero_spatial amplify_checkpoint=checkpoints/AMPLIFY/libero_zero_shot.pt
python eval_libero.py dataset=[libero_goal] run_name=libero_zero_shot_libero_goal amplify_checkpoint=checkpoints/AMPLIFY/libero_zero_shot.pt