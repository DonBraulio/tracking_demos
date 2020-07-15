# %%
import os
import sys
import torch
import cv2
import numpy as np

from tqdm import tqdm

# Imports from AlphaPose subtree
ALPHAPOSE_DIR = "../AlphaPose"
sys.path.append(ALPHAPOSE_DIR)
from SPPE.src.main_fast_inference import InferenNet, InferenNet_fast  # noqa
from opt import opt  # noqa
from fn import getTime  # noqa
from dataloader import Mscoco, VideoLoader  # noqa

# Modified from Alphapose/ implementation
from integration_alphapose.custom_dataloader import (
    DetectionLoader,
    DetectionProcessor,
    DataWriter,
)  # noqa

sys.path.append("../norfair")
import norfair  # noqa


# %%
def alphapose_process_video(videofile, pose_result_handler, inference_steps=1):
    if not len(videofile):
        raise IOError("Error: must contain --video")

    # Load input video
    print(f"Opening video {videofile}")
    data_loader = VideoLoader(videofile, batchSize=args.detbatch).start()
    (fourcc, fps, frameSize) = data_loader.videoinfo()

    # Load detection loader
    print("Loading YOLO model..")
    sys.stdout.flush()
    det_loader = DetectionLoader(
        data_loader, batchSize=args.detbatch, path=ALPHAPOSE_DIR, inference_steps=inference_steps,
    ).start()
    det_processor = DetectionProcessor(det_loader).start()

    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset, path=ALPHAPOSE_DIR)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset, path=ALPHAPOSE_DIR)
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {"dt": [], "pt": [], "pn": []}

    # Data writer
    args.save_video = args.video_savefile is not None
    writer = DataWriter(
        save_video=args.save_video,  # Note: DataWriter uses args.save_video internally as well
        savepath=args.video_savefile,
        fourcc=cv2.VideoWriter_fourcc(*"XVID"),
        fps=fps,
        frameSize=frameSize,
        result_handler=pose_result_handler,
    ).start()

    im_names_desc = tqdm(range(data_loader.length()))
    batchSize = args.posebatch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            if orig_img is None:
                break
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split("/")[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile["dt"].append(det_time)
            # Pose Estimation

            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize : min((j + 1) * batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile["pt"].append(pose_time)

            hm = hm.cpu().data
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split("/")[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile["pn"].append(post_time)

        if args.profile:
            # TQDM
            im_names_desc.set_description(
                "det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}".format(
                    dt=np.mean(runtime_profile["dt"]),
                    pt=np.mean(runtime_profile["pt"]),
                    pn=np.mean(runtime_profile["pn"]),
                )
            )

    print("===========================> Finish Model Running.")
    if (args.save_img or args.video_savefile) and not args.vis_fast:
        print("===========================> Rendering remaining images in the queue...")
        print(
            "===========================> If this step takes too long, you can enable "
            "the --vis_fast flag to use fast rendering (real-time)."
        )
    while writer.running():
        pass
    writer.stop()
    return writer.results()


def convert_and_filter(poses):
    idx_shoulders = [5, 6]
    idx_hip = [11, 12]
    detections = []
    for pose in poses:
        scores = pose["kp_score"].flatten()
        if (scores[idx_shoulders] < 0.05).all() or (scores[idx_hip] < 0.05).all():
            continue
        pos_neck = pose["keypoints"][idx_shoulders, :2].numpy().mean(axis=0).astype(int)
        pos_hip = pose["keypoints"][idx_hip, :2].numpy().mean(axis=0).astype(int)
        detections.append(np.array([pos_neck, pos_hip]))
    return detections


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


tracker = norfair.Tracker(distance_function=keypoints_distance)


def tracker_update(frame, result):
    # AlphaPose result
    detected_poses = result["result"]

    converted_detections = convert_and_filter(detected_poses)
    predictions = tracker.update(converted_detections, dt=1)
    norfair.draw_points(frame, converted_detections)

    norfair.draw_predictions(frame, predictions)

    # Draw on a copy of the frame
    # frame_processed = frame_img.copy()
    # for pose in poses:
    #     pos_nose = pose["keypoints"][0, :2].int().numpy()
    #     cv2.rectangle(
    #         frame_processed, tuple(pos_nose - (2, 2)), tuple(pos_nose + (2, 2)), (0, 0, 255), 1
    #     )

    return frame


# %%
# Reset kernel if GPU is already in use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Alphapose arguments
args = opt
args.dataset = "coco"
args.sp = True
args.outputpath = "./outputs"
if args.outputpath and not os.path.exists(args.outputpath):
    os.mkdir(args.outputpath)

for video_name in ["demo2_10s"]:  # , "demo2_10s"]:
    video_inputfile = f"./{video_name}.mp4"
    args.video_savefile = os.path.join(args.outputpath, f"output_{video_name}_bit.avi")

    # Process video
    alphapose_process_video(
        video_inputfile, tracker_update, inference_steps=1,
    )

# %%
