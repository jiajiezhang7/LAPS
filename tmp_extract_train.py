import os, glob, subprocess, sys
IN='/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_train.split1_cam01'
OUT='/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames_train'
LOG='/home/johnny/action_ws/output/otas_logs/step4_extract_train.log'
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.dirname(LOG), exist_ok=True)
open(LOG,'w').close()
avi_files = sorted(glob.glob(os.path.join(IN, '*.avi')))
processed=skipped=failed=0
for v in avi_files:
    bn = os.path.basename(v)[:-4]
    p = bn.split('_')[0]
    cam = bn.split('_')[1]
    rest = '_'.join(bn.split('_')[2:])
    if rest.startswith(p + '_'):
        rest = rest[len(p)+1:]
    dest = os.path.join(OUT, f'{p}_{cam}_{rest}')
    os.makedirs(dest, exist_ok=True)
    has_frames = any(name.startswith('Frame_') and name.lower().endswith('.jpg') for name in os.listdir(dest))
    if has_frames:
        with open(LOG,'a') as f: f.write(f'[SKIP] {bn} existing\n'); skipped+=1; continue
    cmd = ['ffmpeg','-hide_banner','-loglevel','error','-y','-i', v, os.path.join(dest,'Frame_%06d.jpg')]
    try:
        rc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if rc.returncode == 0:
            with open(LOG,'a') as f: f.write(f'[OK] {bn}\n'); processed+=1
        else:
            with open(LOG,'a') as f: f.write(f'[FAIL] {bn}: {rc.stderr.decode(utf-8,errors=ignore)[:200]}\n'); failed+=1
    except Exception as e:
        with open(LOG,'a') as f: f.write(f'[FAIL] {bn}: {e}\n'); failed+=1
with open(LOG,'a') as f: f.write(f'[DONE] processed={processed} skipped={skipped} failed={failed}\n')
print(LOG)
