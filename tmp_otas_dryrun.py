import os, sys
sys.path.insert(0, '/home/johnny/action_ws/comapred_algorithm/OTAS/code')
from arg_pars import opt
from dataset import Breakfast
opt.frame_path = '/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames_test'
opt.video_info_file = '/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/video_info_test.pkl'
opt.dataset = 'BF'
opt.feature_model = 'tf'
opt.view = 'cam01'
opt.batch_size = 8
opt.num_workers = 4
view_suffix = '_' + opt.view if hasattr(opt, 'view') and len(opt.view) > 0 else ''
opt.pkl_folder_name = opt.output_path + 'OTAS/' + opt.dataset+'_'+opt.feature_model + view_suffix
print('[INFO] pkl_folder_name =', opt.pkl_folder_name)
print('[INFO] constructing Breakfast(val) ...')
val_ds = Breakfast(seq_len=opt.seq_len, num_seq=opt.num_seq, downsample=opt.ds, pred_step=opt.pred_step, mode='val', view_filter=opt.view)
print('[RESULT] dataset size (val) =', len(val_ds))
