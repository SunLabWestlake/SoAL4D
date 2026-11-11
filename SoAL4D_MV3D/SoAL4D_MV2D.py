import os
import cv2
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
tf.config.experimental.set_memory_growth(gpu[1], True)
tf.config.experimental.set_memory_growth(gpu[2], True)

from hyperpose import Config,Model,Dataset
from RT3D_free.utils import check_path, get_all_img_names, name2info, scale_keypoint, resign_frame_id, user_def_color, \
    rgb_color
from RT3D_free.rotate import judge_visible_for_two_flies,distinguish_2_flies, change_visible

SOAL4D_MV2D_INPUT_IMAGE_PATH = r"../data/soal_hyper/exp_2flies/images"
SOAL4D_MV2D_LATERAL_WEIGHT_PATH = r"models\USERDEF_Openpose_Resnet18_20220323_1543_free_side_3\model_dir\newest_model.npz"
SOAL4D_MV2D_TOP_WEIGHT_PATH = r"models\USERDEF_Openpose_Resnet18_20220125_2007_free_top_2\model_dir\newest_model - 9000.npz"

def creat_model():
    model_type = "Openpose"
    model_backbone = "Resnet18"

    dataset_type = "USERDEF"
    model_name = dataset_type+'_'+model_type+'_' + model_backbone + '_'+time.strftime("%Y%m%d_%H%M")
    dataset_version ="2021"
    dataset_path = "./data"
    train_type = "Single_train"
    kf_optimizer ='Sync_avg'
    userdef_dataset = 1
    useradd_data_path = "./data"
    domainadapt_data_path = None

    optim_type ="Adam"
    learning_rate =1e-4
    log_interval =1e2
    save_interval =5e3

    Config.set_model_name(model_name)
    Config.set_model_type(Config.MODEL[model_type])
    Config.set_model_backbone(Config.BACKBONE[model_backbone])
    Config.set_log_interval(log_interval)
    Config.set_save_interval(save_interval)
    # config train
    Config.set_train_type(Config.TRAIN[train_type])
    Config.set_learning_rate(learning_rate)
    Config.set_optim_type(Config.OPTIM[optim_type])
    Config.set_kungfu_option(Config.KUNGFU[kf_optimizer])

    # config dataset
    print(f"test enabling dataset:{userdef_dataset}")
    Config.set_userdef_dataset(userdef_dataset)
    Config.set_dataset_type(Config.DATA[dataset_type])
    Config.set_dataset_path(dataset_path)
    Config.set_dataset_version(dataset_version)

    # train
    config = Config.get_config()

    config.pretrain.batch_size = 2
    config.eval.batch_size = 2
    config.train.batch_size = 2

    config.model.hin = 480
    config.model.win = 640
    config.model.hout = 60
    config.model.wout = 80
    config.train.save_interval = 5000
    config.train.n_step = 30000

    print(config.model)
    print(config.train)
    print(config.data)

    model = Model.get_model(config)
    return model

def infer_one_image(model,img,postprocess=None):
    if postprocess==None:
        postprocess = Model.get_postprocess(Config.MODEL.Openpose)
    img_resized = cv2.resize(np.copy(img), (model.win, model.hin))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_image = img_rgb.astype(np.float32) / 255.0
    if (model.data_format == "channels_first"):
        input_image = np.transpose(input_image, [2, 0, 1])
    img_h, img_w, img_c= input_image.shape
    conf_map, paf_map = model.infer(input_image[np.newaxis, :, :, :]) # TensorShape([1, 60, 80, 42]), TensorShape([1, 60, 80, 88])
    humans = postprocess(conf_map[0], paf_map[0], img_h, img_w, model.parts, model.limbs, model.data_format,
                         model.colors)
    return humans

def infer_one_image_by_conf_map(model, img): # for single
    num_joint = 41
    keypoints = np.zeros([num_joint,3])

    img_resized = cv2.resize(np.copy(img), (model.win, model.hin))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_image = img_rgb.astype(np.float32) / 255.0
    if (model.data_format == "channels_first"):
        input_image = np.transpose(input_image, [2, 0, 1])
    img_h, img_w, img_c= input_image.shape
    conf_map, paf_map = model.infer(input_image[np.newaxis, :, :, :]) # TensorShape([1, 60, 80, 42]), TensorShape([1, 60, 80, 88])

    conf_map_head = np.transpose(conf_map[0], [2, 0, 1])
    output_image = np.copy(input_image)
    for joint_id in range(41):
        conf_head = np.abs(conf_map_head[joint_id, :, :])
        heatmap_avg = cv2.resize(conf_head,
                                 (input_image.shape[1], input_image.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)
        y, x = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
        keypoints[joint_id, :] = [x, y, 2]
    return keypoints

def humans_2_keypoints_for_single(humans):
    # result = []
    result = [[] for idx in range(0, 41)]
    for human in humans:
        # print('\n')
        keypoints = np.zeros([41, 3])
        for part_idx in human.body_parts.keys():
            body_part = human.body_parts[part_idx]
            x = body_part.x # * 8
            y = body_part.y # * 8
            score = body_part.score
            # color = colors[part_idx]
            # color = rgb_color(user_def_color(part_idx), if_255=True)
            # print('\t\t',part_idx, x, y)
            # keypoints[part_idx] = [int(x), int(y), score]
            # result.append(keypoints)
            result[part_idx].append([int(x), int(y), score])
    keypoints = np.zeros([41, 3])
    for joint_id, point in enumerate(result):
        if len(point)==1:
            keypoints[joint_id] = point[0] #[int(x), int(y), 2]
        if len(point)>1:
            # print('\t',joint_id)
            keypoints[joint_id] = choose_point(point)
            # print(joint_id, point)
            # print(choose_point(point))
    return keypoints

def choose_point(point):
    max_score = 0.2
    max_score_id = 0
    for idx, (x,y,score) in enumerate(point):
        print('\t\t',score)
        if score > max_score:
            max_score_id = idx
            max_score = score
    if max_score>0.1:
        return point[max_score_id]
    else:
        point[max_score_id][2] = 0
        return point[max_score_id]

def infer_images_social(path_from, weight_path_side, weight_path_top):
    model = creat_model()
    img_names = get_all_img_names(path_from)

    path_to = os.path.join(path_from, 'output')
    check_path(path_to)

    path_to_image = os.path.join(path_from, 'output', 'image')
    check_path(path_to_image)

    num_pic = len(img_names)
    num_cam = 9
    num_joint = 41
    pose2d_result = np.zeros([num_cam, int(num_pic/9), num_joint, 3])
    print(pose2d_result.shape)

    frame_ids_new_all = resign_frame_id(path_from)
    start_time = time.time()
    weight_path = None
    for idx, img_name in enumerate(img_names): # img_name=img_names[8]
        fly_id, cam_id, frame_id = name2info(img_name)
        frame_id_new = frame_ids_new_all.index(frame_id)

        weight_path1 = weight_path_top if cam_id == 8 else weight_path_side
        if weight_path1 != weight_path:
            model.load_weights(weight_path1)
        weight_path = weight_path1
        print(f'load {weight_path}')

        print(img_name)
        img = cv2.imread(os.path.join(path_from, img_name))

        if cam_id == 8:
            humans = infer_one_image(model, img)
            if len(humans)==0:
                keypoints = np.zeros([41, 3])
            else:
                keypoints_all = humans_2_keypoints_for_social(humans, threshold=7)  # PAF threshold
                if len(keypoints_all) >= 2:
                    keypoints, keypoints_1 = distinguish_2_flies(keypoints_all)
                    cam_occlued_id = judge_visible_for_two_flies(keypoints_all)
                    print('cam_occlued', cam_occlued_id)
                    pose2d_result[cam_occlued_id, frame_id_new, :, :] = 0

                keypoints_all = humans_2_keypoints_for_social(humans, threshold=13)
                if len(keypoints_all) >= 2:
                    keypoints, keypoints_1 = distinguish_2_flies(keypoints_all)
                elif len(keypoints_all) == 1:
                    keypoints = keypoints_all[0]
        elif cam_id < 8:
            humans = infer_one_image(model, img)
            try:
                keypoints_all = humans_2_keypoints_for_social(humans, threshold=13)
                assert len(keypoints_all) != 0
                sum_score = np.sum(keypoints_all[:, :, 2], axis=1)
                keypoints = keypoints_all[np.where(sum_score == np.max(sum_score))][0]
            except:
                keypoints = np.zeros([41,3])
        frame_id_new = frame_ids_new_all.index(frame_id)
        pose2d_result[cam_id, frame_id_new] = keypoints
    return pose2d_result

def humans_2_keypoints_for_social(humans,threshold=13):
    keypoints_all_temp = []
    for human in humans:
        keypoints = np.zeros([41, 3])
        for part_idx in human.body_parts.keys():
            body_part = human.body_parts[part_idx]
            x = body_part.x
            y = body_part.y
            score = body_part.score
            keypoints[part_idx] = [int(x), int(y), score]
        keypoints_all_temp.append(keypoints)
    keypoints_all_np = np.asarray(keypoints_all_temp)
    sum_score = np.sum(keypoints_all_np[:, :, 2], axis=1)
    print(sum_score)
    keypoints_all = keypoints_all_np[np.where(sum_score >= threshold)]
    return keypoints_all

def infer_social(path_from, weight_path_side, weight_path_top):
    check_path(os.path.join(path_from, 'output'))
    pose2d_result = infer_images_social(path_from, weight_path_side, weight_path_top)
    pose2d_result = scale_keypoint(pose2d_result)
    pose_result_np = change_visible(pose2d_result)

    pkl_name = f'2d_predicted_visualized.pkl'
    pkl_path = os.path.join(path_from, 'output', pkl_name)
    with open(pkl_path, "wb") as f:
        print(pkl_path)
        pickle.dump(pose_result_np, f, pickle.DEFAULT_PROTOCOL)  # 4
        print(pose_result_np.shape)
    plt.close('all')

def draw_keypoints(keypoints,img_rgb):
    img_h, img_w, img_c = img_rgb.shape
    radius = max(int(min(img_h, img_w) / 100), 1)
    thickness = max(int(min(img_h, img_w) / 120), 1)
    output_image = np.copy(img_rgb)
    for joint_id, joint in enumerate(keypoints):
        x, y, score = joint
        color = rgb_color(user_def_color(joint_id), if_255=True)
        # print(joint_id, x, y, color)
        if score !=0:
            output_image = cv2.circle(output_image, (int(x), int(y)), radius=radius, color=color, thickness=-1)
        # img = cv2.putText(img, str(part_idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
        #                   0.3, color, 1, cv2.LINE_AA)

    # plt.figure()
    # plt.imshow(output_image)
    return output_image

if __name__ == '__main__':
    infer_social(SOAL4D_MV2D_INPUT_IMAGE_PATH, SOAL4D_MV2D_LATERAL_WEIGHT_PATH, SOAL4D_MV2D_TOP_WEIGHT_PATH)

