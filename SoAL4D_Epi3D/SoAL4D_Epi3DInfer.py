'''
    input: the image cropped from MV2D
    output: the 3D pose predicted from Epi3D
'''
import os
import cv2
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import matplotlib

import epipose.models as models
from epipose.core.config import config
from epipose.core.config import update_config
from epipose.core.integral_loss import get_joint_location_result
from epipose.utils.img_utils import convert_cvimg_to_tensor
from RT3D_free.utils import rgb_color, user_def_color, info2name, check_path, name2info, get_all_img_names

from soal_analysis.path_config import get_ht_roi_img, get_pa_by_frame, get_pa_by_frame_batch, get_kpt_by_frame_batch

SOAL4D_EPI3D_GPU = [2]  # [0,1,2] for server
# torch.cuda.set_per_process_memory_fraction(0.6, 0)

SOAL4D_EPI3D_PREPROC_OUTPUT = r'..\data\preproc_1m1f'
SOAL4D_EPI3D_FINAL_WEIGHT_PATH = "models/model_best.pth.tar"

matplotlib.get_backend()
matplotlib.use('Agg')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

cfg_file = '../epipose/config_rt3d/valid.yaml'
update_config(cfg_file)
config.MODEL.RESUME = SOAL4D_EPI3D_FINAL_WEIGHT_PATH

torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED

image_size = config.MODEL.IMAGE_SIZE[0]

model = models.pose3d_resnet.get_pose_net(config, is_train=False)
print(f'Epi3D Using GPU {SOAL4D_EPI3D_GPU}...')
model = torch.nn.DataParallel(model, device_ids=SOAL4D_EPI3D_GPU).to("cuda:%d"%(SOAL4D_EPI3D_GPU[0]))
print('Epi3D Created model...')

checkpoint = torch.load(config.MODEL.RESUME)
model.load_state_dict(checkpoint)
model.eval()
print('Epi3D Loaded pretrained weights...')

colors = []
for K in range(42):
    colors.append(rgb_color(user_def_color(K)))

def select_interval(num_from,num_to=200):
    interval = 1
    while num_from/interval>num_to:
        interval *=2
    return interval

bone_part = [[11, 12], [12, 13], [13, 14], [14, 15], [16, 17], [17, 18], [18, 19], [19, 20], [21, 22], [22, 23], [23, 24], [24, 25],
             [26, 27], [27, 28], [28, 29], [29, 30], [31, 32], [32, 33], [33, 34], [34, 35], [36, 37], [37, 38], [38, 39], [39, 40]]
def plot_top_view_pose_3d_for_epipose(img, pose_3d):
    plt.imshow(img, 'gray')
    plt.scatter(pose_3d[:, 0], pose_3d[:, 1], c=colors[:41])
    plt.scatter(pose_3d[11:26, 2] + 30, pose_3d[11:26, 1], s=4, c=colors[11:26])
    plt.scatter(226 - pose_3d[26:, 2], pose_3d[26:, 1], s=4, c=colors[26:41])
    for jid in range(40):
        if [jid, jid+1] in bone_part:
            plt.plot(pose_3d[jid:(jid+2), 0], pose_3d[jid:(jid+2), 1], c=colors[jid])

def _predict(image):
    img_patch = convert_cvimg_to_tensor(image)
    # img_patch = torch.from_numpy(img_patch).cuda()
    img_patch = torch.from_numpy(img_patch).to(device="cuda:%d"%(SOAL4D_EPI3D_GPU[0]))
    preds_org = model(img_patch[None, ...])  # 1 3 256 256
    preds = get_joint_location_result(image_size, image_size, preds_org)
    preds = preds[0, :, :3]
    return preds

def predict_for_test(img_path,debug_mode=False):
    image = cv2.imread(img_path)
    if 256 != image.shape[0]:
        image = cv2.resize(image,(256,256))
    pose_3d = _predict(image)

    if debug_mode:
        img = image[:,:,0].copy()
        plt.figure(figsize=(12,7))
        plt.subplot(1,2,1)
        plt.imshow(img,'gray')

        plt.subplot(1,2,2)
        plt.title('3d epipose')
        plot_top_view_pose_3d_for_epipose(img, pose_3d)
    return pose_3d

def predict_batch_img(batch_l, debug_mode=False):
    img_patch_l = [convert_cvimg_to_tensor(b[-1]) for b in batch_l]  # batch_size 3 256 256
    batch = torch.from_numpy(np.array(img_patch_l)).to(device="cuda:%d"%(SOAL4D_EPI3D_GPU[0]))
    preds_org = model(batch)
    preds = get_joint_location_result(image_size, image_size, preds_org)
    preds = preds[:, :, :3]

    if debug_mode:  # plot last one
        img = img[:,:,0].copy()
        plt.figure(figsize=(12,7))
        plt.subplot(1,2,1)
        plt.imshow(img,'gray')

        plt.subplot(1,2,2)
        plt.title('3d epipose')
        plot_top_view_pose_3d_for_epipose(img, preds[-1])

        plt.savefig(path.replace(".jpg", "_preview.jpg"))
        plt.close('all')
    return preds

def preview_pred_npy(npy, path_to):
    if not npy.endswith("npy"):
        return
    print("preview", npy)
    name = os.path.basename(npy) #fly_1203_cam_8_img_000155.npy
    tt = name[:-4].split("_")
    fly_id = int(tt[1])
    frame = int(tt[5])
    fly_idx = int(os.path.basename(os.path.dirname(npy)))
    img = get_ht_roi_img(fly_id, frame)
    preds = np.load(npy, allow_pickle=True)

    fly_pa = get_pa_by_frame(fly_id, frame, fly_idx)
    if fly_pa is None:
        print("get_pa_by_frame fail")
        return
    from SoAL4D_Epi3D.SoAL4D_Epi3DPrepro import crop_and_rotate_img_by_GPU
    img_final = crop_and_rotate_img_by_GPU(img[:,:,0], fly_pa, fly_pa[11])

    plt.figure(figsize=(6, 7))
    plot_top_view_pose_3d_for_epipose(img_final, preds)
    jpg = os.path.join(path_to, name.replace(".npy", "_preview.jpg"))
    plt.savefig(jpg)
    print("save preview", jpg)

def preview_epi(fly_id, frame, fly_idx, path_to):
    img = get_ht_roi_img(fly_id, frame)
    fly_pa = get_pa_by_frame_batch(fly_id, frame, fly_idx)
    if fly_pa is None:
        print("get_pa_by_frame fail", fly_id, frame, fly_idx)
        return
    fly_kpt = get_kpt_by_frame_batch(fly_id, frame, fly_idx)   # 41, 3
    if fly_kpt is None:
        print("get_kpt_by_frame fail", fly_id, frame, fly_idx)
        return

    if fly_pa[11] == 0:
        print("not all fly detected", fly_id, frame, fly_idx)
        return
    from SoAL4D_Epi3D.SoAL4D_Epi3DPrepro import crop_and_rotate_img_by_GPU
    img_final = crop_and_rotate_img_by_GPU(img[:,:,0], fly_pa, fly_pa[11])

    plt.figure(figsize=(6, 7))
    plot_top_view_pose_3d_for_epipose(img_final, fly_kpt)
    name = "kpt_%06d_%06d_%d.jpg" % (frame, fly_id, fly_idx)
    jpg = os.path.join(path_to, name)
    plt.savefig(jpg)
    print("save preview", jpg)

def predict_batch(batch_l, path_to, debug_mode=False):
    img_patch_l = []
    for img_path in batch_l:
        image = cv2.imread(img_path)
        if 256 != image.shape[0]:
            image = cv2.resize(image, (256, 256))
        img_patch = convert_cvimg_to_tensor(image)
        img_patch = torch.from_numpy(img_patch).cuda()
        img_patch_l.append(img_patch)
    batch = torch.stack(img_patch_l)  # batch_size 3 256 256
    preds_org = model(batch)
    preds = get_joint_location_result(image_size, image_size, preds_org)
    preds = preds[:, :, :3]

    if debug_mode:
        img = image[:,:,0].copy()
        plt.figure(figsize=(12,7))
        plt.subplot(1,2,1)
        plt.imshow(img,'gray')

        plt.subplot(1,2,2)
        plt.title('3d epipose')
        plot_top_view_pose_3d_for_epipose(img, preds[-1])

        img_path_new = os.path.join(path_to, 'preview', os.path.basename(img_path).replace('.npy', ''))
        plt.savefig(img_path_new)
        plt.close('all')
    return preds

def get_all_npy_names(path_from):
    img_names = os.listdir(path_from)
    img_names = [img_name for img_name in img_names if img_name.endswith('.npy')]
    print(f'The fold {path_from} contains {len(img_names)} images')
    return img_names

def main(path_from, start_frame=0, batch_preview_inter=50):
    path_to = path_from + '_output'
    check_path(path_to)
    path_to_1 = os.path.join(path_to, 'preview')
    check_path(path_to_1)

    interval = 1
    npy_names = get_all_npy_names(path_from+'_output')
    img_names = get_all_img_names(path_from)
    print('img_names', len(img_names))
    print('npy_names', len(npy_names))

    if len(img_names) > 1000:
        save_interval_fold = 100
    else:
        num_from = len(img_names)
        save_interval_fold = select_interval(num_from, num_to=200)

    time_start = time.time()
    batch_l = []
    batch_size = 10
    batch_i = 0
    n_imgs = len(img_names)
    for img_i in range(n_imgs):
        img_name = img_names[img_i]
        fly_id, cam_id, frame_id = name2info(img_name)
        if frame_id < start_frame:
            continue
        if frame_id % interval != 0:
            continue
        npy_path = os.path.join(path_to, img_name.replace('.jpg', '.npy'))
        if os.path.exists(npy_path):
            continue

        img_path = os.path.join(path_from, img_name)
        # preds = predict_for_test(img_path)
        # np.save(os.path.join(path_to, img_name.replace('.jpg', '.npy')), preds)
        if len(batch_l) >= batch_size or img_i == n_imgs-1:
            batch_l.append(img_path)
            predss = predict_batch(batch_l, path_to, debug_mode=batch_i%batch_preview_inter==0)
            for img_path1, preds in zip(batch_l, predss):
                img_name1 = os.path.basename(img_path1)
                np.save(os.path.join(path_to, img_name1.replace('.jpg', '.npy')), preds)
            batch_l = []
            batch_i += 1
        else:
            batch_l.append(img_path)

        if frame_id % (10 * interval) == 0:
            print(img_name)
        # if frame_id % (save_interval_fold * interval+1) == 0:
        #     preds = predict_for_test(img_path,debug_mode=True)
        #     img_path_new = os.path.join(path_to, 'preview', img_name.replace('.npy', ''))
        #     plt.savefig(img_path_new)
        #     plt.close('all')

    time_end = time.time()
    print(time_end-time_start)
    print(path_from, '\n')
    print('finish***********\n' * 3)

if __name__ == '__main__':
    for K in range(1, 4):
        path_from = os.path.join(SOAL4D_EPI3D_PREPROC_OUTPUT, str(K))
        print(path_from)
        main(path_from)
