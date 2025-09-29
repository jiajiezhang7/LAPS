# Motion Tokenizer
python train_motion_tokenizer.py run_name=libero_10_motion_tokenizer train_datasets=[libero_10:traj1.0] val_datasets=[libero_10:traj-0.1]
python train_motion_tokenizer.py run_name=libero_object_motion_tokenizer train_datasets=[libero_object:traj1.0] val_datasets=[libero_object:traj-0.1]
python train_motion_tokenizer.py run_name=libero_spatial_motion_tokenizer train_datasets=[libero_spatial:traj1.0] val_datasets=[libero_spatial:traj-0.1]
python train_motion_tokenizer.py run_name=libero_goal_motion_tokenizer train_datasets=[libero_goal:traj1.0] val_datasets=[libero_goal:traj-0.1]
python train_motion_tokenizer.py run_name=libero_90_motion_tokenizer train_datasets=[libero_90:traj1.0] val_datasets=[libero_90:traj-0.1]

# Forward Dynamics (uses Motion Tokenizer checkpoints)
python train_forward_dynamics.py run_name=libero_10_forward_dynamics forward_dynamics.motion_tokenizer.checkpoint=checkpoints/motion_tokenizer/libero_10_motion_tokenizer/latest.pt train_datasets=[libero_10:traj1.0] val_datasets=[libero_10:traj-0.1]
python train_forward_dynamics.py run_name=libero_object_forward_dynamics forward_dynamics.motion_tokenizer.checkpoint=checkpoints/motion_tokenizer/libero_object_motion_tokenizer/latest.pt train_datasets=[libero_object:traj1.0] val_datasets=[libero_object:traj-0.1]
python train_forward_dynamics.py run_name=libero_spatial_forward_dynamics forward_dynamics.motion_tokenizer.checkpoint=checkpoints/motion_tokenizer/libero_spatial_motion_tokenizer/latest.pt train_datasets=[libero_spatial:traj1.0] val_datasets=[libero_spatial:traj-0.1]
python train_forward_dynamics.py run_name=libero_goal_forward_dynamics forward_dynamics.motion_tokenizer.checkpoint=checkpoints/motion_tokenizer/libero_goal_motion_tokenizer/latest.pt train_datasets=[libero_goal:traj1.0] val_datasets=[libero_goal:traj-0.1]
python train_forward_dynamics.py run_name=libero_90_forward_dynamics forward_dynamics.motion_tokenizer.checkpoint=checkpoints/motion_tokenizer/libero_90_motion_tokenizer/latest.pt train_datasets=[libero_90:traj1.0] val_datasets=[libero_90:traj-0.1]

# Inverse Dynamics (uses Motion Tokenizer and Forward Dynamics checkpoints)
python train_inverse_dynamics.py run_name=libero_10_inverse_dynamics motion_tokenizer_checkpoint=checkpoints/motion_tokenizer/libero_10_motion_tokenizer/latest.pt forward_dynamics_checkpoint=checkpoints/forward_dynamics/libero_10_forward_dynamics/latest.pt train_datasets=[libero_10:action1.0] val_datasets=[libero_10:action-0.1]
python train_inverse_dynamics.py run_name=libero_object_inverse_dynamics motion_tokenizer_checkpoint=checkpoints/motion_tokenizer/libero_object_motion_tokenizer/latest.pt forward_dynamics_checkpoint=checkpoints/forward_dynamics/libero_object_forward_dynamics/latest.pt train_datasets=[libero_object:action1.0] val_datasets=[libero_object:action-0.1]
python train_inverse_dynamics.py run_name=libero_spatial_inverse_dynamics motion_tokenizer_checkpoint=checkpoints/motion_tokenizer/libero_spatial_motion_tokenizer/latest.pt forward_dynamics_checkpoint=checkpoints/forward_dynamics/libero_spatial_forward_dynamics/latest.pt train_datasets=[libero_spatial:action1.0] val_datasets=[libero_spatial:action-0.1]
python train_inverse_dynamics.py run_name=libero_goal_inverse_dynamics motion_tokenizer_checkpoint=checkpoints/motion_tokenizer/libero_goal_motion_tokenizer/latest.pt forward_dynamics_checkpoint=checkpoints/forward_dynamics/libero_goal_forward_dynamics/latest.pt train_datasets=[libero_goal:action1.0] val_datasets=[libero_goal:action-0.1]
python train_inverse_dynamics.py run_name=libero_90_inverse_dynamics motion_tokenizer_checkpoint=checkpoints/motion_tokenizer/libero_90_motion_tokenizer/latest.pt forward_dynamics_checkpoint=checkpoints/forward_dynamics/libero_90_forward_dynamics/latest.pt train_datasets=[libero_90:action1.0] val_datasets=[libero_90:action-0.1]

# Bundle
python -m amplify.bundle_amplify --mt_ckpt checkpoints/motion_tokenizer/libero_10_motion_tokenizer/latest.pt --fd_ckpt checkpoints/forward_dynamics/libero_10_forward_dynamics/latest.pt --id_ckpt checkpoints/inverse_dynamics/libero_10_inverse_dynamics_seed_0/latest.pt --name libero_10_bc
python -m amplify.bundle_amplify --mt_ckpt checkpoints/motion_tokenizer/libero_90_motion_tokenizer/latest.pt --fd_ckpt checkpoints/forward_dynamics/libero_90_forward_dynamics/latest.pt --id_ckpt checkpoints/inverse_dynamics/libero_90_inverse_dynamics_seed_0/latest.pt --name libero_90_bc
python -m amplify.bundle_amplify --mt_ckpt checkpoints/motion_tokenizer/libero_object_motion_tokenizer/latest.pt --fd_ckpt checkpoints/forward_dynamics/libero_object_forward_dynamics/latest.pt --id_ckpt checkpoints/inverse_dynamics/libero_object_inverse_dynamics_seed_0/latest.pt --name libero_object_bc
python -m amplify.bundle_amplify --mt_ckpt checkpoints/motion_tokenizer/libero_spatial_motion_tokenizer/latest.pt --fd_ckpt checkpoints/forward_dynamics/libero_spatial_forward_dynamics/latest.pt --id_ckpt checkpoints/inverse_dynamics/libero_spatial_inverse_dynamics_seed_0/latest.pt --name libero_spatial_bc
python -m amplify.bundle_amplify --mt_ckpt checkpoints/motion_tokenizer/libero_goal_motion_tokenizer/latest.pt --fd_ckpt checkpoints/forward_dynamics/libero_goal_forward_dynamics/latest.pt --id_ckpt checkpoints/inverse_dynamics/libero_goal_inverse_dynamics_seed_0/latest.pt --name libero_goal_bc

# Eval
python eval_libero.py dataset=[libero_10] run_name=libero_10_bc amplify_checkpoint=checkpoints/AMPLIFY/libero_10_bc.pt
python eval_libero.py dataset=[libero_90] run_name=libero_90_bc amplify_checkpoint=checkpoints/AMPLIFY/libero_90_bc.pt n_envs=1
python eval_libero.py dataset=[libero_object] run_name=libero_object_bc amplify_checkpoint=checkpoints/AMPLIFY/libero_object_bc.pt
python eval_libero.py dataset=[libero_spatial] run_name=libero_spatial_bc amplify_checkpoint=checkpoints/AMPLIFY/libero_spatial_bc.pt
python eval_libero.py dataset=[libero_goal] run_name=libero_goal_bc amplify_checkpoint=checkpoints/AMPLIFY/libero_goal_bc.pt