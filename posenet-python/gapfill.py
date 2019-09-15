import tensorflow as tf
import cv2
import time
import argparse
import os

import posenet
import json
import numpy as np

MEMORY = 2
t_coeffs = [0.5**(MEMORY - m - (m == 0)) for m in range(MEMORY)]

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--json_dir', type=str, default='./keypoints')
args = parser.parse_args()


def apply_smoothing(prev_scores, prev_coords, curr_scores, curr_coords):
    out_scores = []
    out_coords = []

    for i in range(17):     # num key points
        mem_scores = [mem_score[i] for mem_score in prev_scores]
        mem_coords = [mem_coord[i] for mem_coord in prev_coords]

        score_sum = sum(mem_scores)
        inv_score_sum = 1 / MEMORY if score_sum == 0 else 1 / sum(mem_scores)
        norm_scores = [score * inv_score_sum for score in mem_scores]
        out_score = sum([t_coeff * score for t_coeff, score in zip(t_coeffs, mem_scores)])

        com = [sum([mem_coords[m][0] for m in range(MEMORY)]) / MEMORY,
            sum([mem_coords[m][1] for m in range(MEMORY)]) / MEMORY]
        out_x = com[0] + sum([t_coeff * norm * (coord[0] - com[0]) for t_coeff, norm, coord in zip(t_coeffs, norm_scores, mem_coords)])
        out_y = com[1] + sum([t_coeff * norm * (coord[1] - com[1]) for t_coeff, norm, coord in zip(t_coeffs, norm_scores, mem_coords)])

        out_scores.append(out_score)
        out_coords.append([out_x, out_y])
    return out_scores, out_coords

def return_max_area_id(keypoint_coords, pose_scores):
    area = -1
    idx = None
    bbox = None
    for pi in range(len(pose_scores)):
        if pose_scores[pi] == 0.:
            break
        pose = keypoint_coords[pi, :, :]
        xmin = np.min(pose[:, 0])
        xmax = np.max(pose[:, 0])
        ymin = np.min(pose[:, 1])
        ymax = np.max(pose[:, 1])
        new_area = (xmax-xmin) * (ymax-ymin)
        if new_area > area:
            area = new_area
            idx = pi
            bbox = [int(xmin), int(xmax), int(ymin), int(ymax)]
    return idx, bbox

def create_bbox(keypoint_coords, pose_scores):
    bbox = []
    for pi in range(len(pose_scores)):
        if pose_scores[pi] == 0.:
            break
        pose = keypoint_coords[pi, :, :]
        xmin = np.min(pose[:, 0])
        xmax = np.max(pose[:, 0])
        ymin = np.min(pose[:, 1])
        ymax = np.max(pose[:, 1])
        bbox.append([int(xmin), int(xmax), int(ymin), int(ymax)])
    return bbox

def get_iou(bb1, bb2):
    #print(bb1, bb2)
    assert bb1[2] < bb1[3]
    assert bb1[0] < bb1[1]
    assert bb2[2] < bb2[3]
    assert bb2[0] < bb2[1]
    x_left = max(bb1[2], bb2[2])
    y_top = max(bb1[0], bb2[0])
    x_right = min(bb1[3], bb2[3])
    y_bottom = min(bb1[1], bb2[1])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[3] - bb1[2]) * (bb1[1] - bb1[0])
    bb2_area = (bb2[3] - bb2[2]) * (bb2[1] - bb2[0])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def main():

    past_coords = [[[0, 0] for _ in range(17)] for __ in range(MEMORY)]
    past_scores = [[0] * 17 for _ in range(MEMORY)]

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        if args.json_dir:
            if not os.path.exists(args.json_dir):
                os.makedirs(args.json_dir)

        filenames = sorted([
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))])

        start = time.time()
        previous_id = None
        previous_bbox = None
        for idx, f in enumerate(filenames):
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

            keypoint_coords *= output_scale
            bboxes = create_bbox(keypoint_coords, pose_scores)
            if previous_id:
                iou = -1
                for key, item in enumerate(bboxes):
                    new_iou = get_iou(item, previous_bbox)
                    iou = max(iou, new_iou)
                    if iou == new_iou:
                        current_id = key
                        current_bbox = item
            else:
                current_id,  current_bbox = return_max_area_id(keypoint_coords, pose_scores)
            
            # smoothing function
            past_coords.append(keypoint_coords[current_id])
            past_coords = past_coords[1:]
            past_scores.append(keypoint_scores[current_id])
            past_scores = past_scores[1:]

            smoothed_scores, smoothed_coords = apply_smoothing(past_scores, past_coords, keypoint_scores[current_id], keypoint_coords[current_id])
            keypoint_scores[current_id] = smoothed_scores
            keypoint_coords[current_id] = smoothed_coords

            previous_id = current_id
            previous_bbox = current_bbox
            if args.output_dir:
                draw_image = posenet.draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.25, min_part_score=0)
                #_ = cv2.rectangle(draw_image, (previous_bbox[2], previous_bbox[0]), (previous_bbox[3], previous_bbox[1]), (0, 0, 255), 2)
                #_ = cv2.rectangle(draw_image, (current_bbox[2], current_bbox[0]), (current_bbox[3], current_bbox[1]), (255, 0, 0), 2)
                #cv2.imshow("out", draw_image)
                #cv2.waitKey(1)
                cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

            data = dict()
            if not args.notxt:
                print()
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    data['Pose%d' % pi] = dict()
                    data['Pose%d' % pi]['score'] = pose_scores[pi]
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                        data['Pose%d' % pi][posenet.PART_NAMES[ki]] = dict()
                        data['Pose%d' % pi][posenet.PART_NAMES[ki]]['score'] = s
                        data['Pose%d' % pi][posenet.PART_NAMES[ki]]['x'] = c[0]
                        data['Pose%d' % pi][posenet.PART_NAMES[ki]]['y'] = c[1]

            with open(os.path.join(args.json_dir, os.path.relpath(f, args.image_dir).split('.')[0]+'.json'), 'w') as fp:
                json.dump(data, fp)

        print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
