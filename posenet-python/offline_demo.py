import tensorflow as tf
import cv2
import time
import argparse
import posenet
import base64
import numpy as np
from collections import deque
from scipy import spatial

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()
#ws = create_connection("ws://66.31.16.203:5000")
whitelist = set('0123456789 .')
bruno_angles = np.load('bruno_angles.npy')

PART_NAMES = ["nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]

limbs = [[(9, 7), (5, 7), (5, 11), (13, 11), (13, 15)],[(10, 8), (6, 8), (6, 12), (14, 12), (14, 16)]]


import math
from scipy import spatial

def angle(v1, v2):
    result = math.atan2(v2[3]-v2[1], v2[2]-v2[0])-math.atan2(v1[3]-v1[1], v1[2]-v1[0])
    if result < -math.pi:
            result += 2*math.pi
    elif result > math.pi:
            result -=2*math.pi
    return result


def get_angles(pose):
    angles = []
    for j in range(2):
        limb = limbs[j]
        for i in range(len(limb)-1):
            limb1 = limb[i] # (9,7)
            limb2 = limb[i+1] # (5,7)
            new = angle([pose[limb1[0]][0], pose[limb1[0]][1], pose[limb1[1]][0], pose[limb1[1]][1]], [pose[limb2[0]][0], pose[limb2[0]][1], pose[limb2[1]][0], pose[limb2[1]][1]])
            angles.append(new)
    return angles

xpeaks = [10,  15,  31,  37,  42,  46,  51,  57,  65,  76,  81,  87,
        91,  97,  99, 101, 105, 108, 110, 112, 114, 119, 121, 125, 135,
       142, 146, 149, 159, 167, 169, 173, 178, 181, 189]

def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        # cv2.imwrite(filename='saved_images/saved_img.jpg', img=cap.read()[1])

        angle_buf = deque()

        start = time.time()
        frame_count = 0
        while True:

            # learn slowly
            i = 0
            while i < len(xpeaks)-1:
                bad_match = True

                curr_segment = xpeaks[i]
                next_segment = xpeaks[i+1]
                
                segment_length = next_segment - curr_segment

                # wait for a secodn for learner to get ready
                bruno = cv2.imread("bruno/%05d.png" % curr_segment)
                cv2.imshow("bruno", bruno)
                cv2.waitKey(1000)

                while bad_match:

                    # start running this segment
                    for frame in range(segment_length):
                        bruno = cv2.imread("bruno/%05d.png" % (curr_segment + frame))

                        input_image, display_image, output_scale = posenet.read_cap(
                            cap, scale_factor=args.scale_factor, output_stride=output_stride)

                        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                            model_outputs,
                            feed_dict={'image:0': input_image}
                        )

                        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                            heatmaps_result.squeeze(axis=0),
                            offsets_result.squeeze(axis=0),
                            displacement_fwd_result.squeeze(axis=0),
                            displacement_bwd_result.squeeze(axis=0),
                            output_stride=output_stride,
                            max_pose_detections=10,
                            min_pose_score=0.15)

                        keypoint_coords *= output_scale

                        # save the angle
                        # turn keypoint_coords into angles
                        angle_data = get_angles(keypoint_coords[0])
                        angle_buf.append(angle_data)
                        if len(angle_buf) > segment_length:
                            angle_buf.popleft()

                            score = 0
                            for i, angle_learn in enumerate(angle_buf):
                                angle_gt = bruno_angles[curr_segment+i]
                                score += 1-spatial.distance.cosine(angle_learn, angle_gt)
                            norm = score / segment_length

                            bad_match = norm < 0.5
                            if not bad_match:
                                print("match!!")
                                i+=1
                                cv2.putText(bruno, "NICE!", (1100, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), thickness=8)
                            print(norm)

                        # shift keypoints to right side of screen
                        for p in range(17):
                            keypoint_coords[0][p][1] = 1280 + (1280 - keypoint_coords[0][p][1])

                        # TODO this isn't particularly fast, use GL for drawing and display someday...
                        overlay_image = posenet.draw_skel_and_kp(
                            bruno, pose_scores, keypoint_scores, keypoint_coords,
                            min_pose_score=0.15, min_part_score=0.1)
                        cv2.imshow("bruno", overlay_image)
                        cv2.waitKey(1)

                cv2.waitKey(500)
                angle_buf = deque()


            print("good game")
            exit()


            ###############################################

         #    # this saves the image file
         #    img = cap.read()[1]
         #    cv2.imwrite(filename='saved_images/img' + str(frame_count) + '.jpg', img=img)
            
         #    with open('saved_images/img' + str(frame_count) + '.jpg', "rb") as f:
         #        data = f.read()

         #    binary_image = base64.encodestring(data)
         #    ws.send(binary_image)
         #    # not sure if this should be here, cuz lag...
         #    result = ws.recv()
         #    # print(result)
         #    pts = result.split('\n\n')[0]
         #    parsed_pts = ''.join(filter(whitelist.__contains__, pts))
         #    lst = parsed_pts.split()
         #    final_lst = [float(f) for f in lst]
         #    keypoints = np.reshape(np.array(final_lst), (17, 2))
         #    cv_keypoints = []
         #    for kc in keypoints:
         #        cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 5. ))
         #    out_img = cv2.drawKeypoints(
	        # img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
         #    	flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
         #    cv2.imshow("out", out_img)
         #    cv2.waitKey(25)
         #    #if cv2.waitKey(1) & 0xFF == ord('q'):
         #    #    break

        #print('Average FPS: ', frame_count / (time.time() - start))
        #ws.close()


if __name__ == "__main__":
    main()
