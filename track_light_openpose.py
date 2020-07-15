import numpy as np
import sys  # For openpose
import os
import torch
import cv2

sys.path.append("../norfair")
import norfair  # noqa

DIR_LIGHT_OPENPOSE = "../lightweight-human-pose-estimation.pytorch"
sys.path.append(DIR_LIGHT_OPENPOSE)
from models.with_mobilenet import PoseEstimationWithMobileNet  # noqa
from modules.keypoints import extract_keypoints, group_keypoints  # noqa
from modules.load_state import load_state  # noqa
from modules.pose import Pose, track_poses  # noqa
from val import normalize, pad_width  # noqa


def convert_and_filter(poses):
    # Get final poses by filtering out parts of the detections we don't want to track
    poses = poses[:, [1, 8]]  # We'll only track neck(1) and midhip(8)
    # Create filter for objects for which we haven't detected any of the parts we want to track
    poses = poses[np.any(poses > 0, axis=(1, 2)), :, :]
    return poses[:, :, :2]  # Remove probabilities from keypoints


def keypoints_distance(detected_pose, person):
    # Find min torax size
    torax_length_detected_person = np.linalg.norm(detected_pose[0] - detected_pose[1])
    predicted_pose = person.prediction
    torax_length_predicted_person = np.linalg.norm(predicted_pose[0] - predicted_pose[1])
    min_torax_size = min(torax_length_predicted_person, torax_length_detected_person)

    # Keypoints distance in terms of torax size
    substraction = detected_pose - predicted_pose
    dists_per_point = np.linalg.norm(substraction, axis=1)
    keypoints_distance = np.mean(dists_per_point) / min_torax_size

    return keypoints_distance


def infer_fast(
    net,
    img,
    net_input_height_size,
    stride,
    upsample_ratio,
    cpu,
    pad_value=(0, 0, 0),
    img_mean=(128, 128, 128),
    img_scale=1 / 256,
):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(
        heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC
    )

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(
        pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC
    )

    return heatmaps, pafs, scale, pad


def pose_detector(img, net, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)
    import ipdb

    # ipdb.set_trace()

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(
            heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
        )

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (
            all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]
        ) / scale
        all_keypoints[kpt_id, 1] = (
            all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]
        ) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)
    return current_poses


def draw_frame(img, detected_poses, track=False):
    orig_img = img.copy()
    for pose in detected_poses:
        pose.draw(frame)
    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    for pose in detected_poses:
        cv2.rectangle(
            img,
            (pose.bbox[0], pose.bbox[1]),
            (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]),
            (0, 255, 0),
        )
        if track:
            cv2.putText(
                img,
                "id: {}".format(pose.id),
                (pose.bbox[0], pose.bbox[1] - 16),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 255),
            )


video = norfair.Video(input_path=sys.argv[1])
tracker = norfair.Tracker(distance_function=keypoints_distance)

# Pose estimation model
net = PoseEstimationWithMobileNet()
checkpoint = torch.load(f"{DIR_LIGHT_OPENPOSE}/checkpoint_iter_370000.pth", map_location="cpu")
load_state(net, checkpoint)

previous_poses = []
for frame in video:
    detected_poses = pose_detector(frame, net, 512, False, False, False)
    # track_poses(previous_poses, detected_poses, smooth=True)
    previous_poses = detected_poses

    draw_frame(frame, detected_poses)

    video.write(frame)
    # video.show(frame, downsample_ratio=1)

os.system(f"ffmpeg -loglevel error -i {video.output_filename} -y {video.output_filename[:-4]}.mp4")
