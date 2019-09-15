import tensorflow as tf
import cv2
import time
import argparse
from websocket import *
import posenet
import base64
import numpy as np

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

        print(cap)

        # cv2.imwrite(filename='saved_images/saved_img.jpg', img=cap.read()[1])

        start = time.time()
        frame_count = 0
        while True:

            # learn slowly
            for i in range(len(xpeaks)-1):
                curr_segment = xpeaks[i]
                next_segment = xpeaks[i+1]
                
                segment_length = next_segment - curr_segment

                # wait for a secodn for learner to get ready
                bruno = cv2.imread("bruno/%05d.png" % curr_segment)
                cv2.waitKey(1000)

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


                    # shift keypoints to right side of screen
                    for p in range(17):
                        keypoint_coords[0][p][1] += 1280

                    # TODO this isn't particularly fast, use GL for drawing and display someday...
                    overlay_image = posenet.draw_skel_and_kp(
                        bruno, pose_scores, keypoint_scores, keypoint_coords,
                        min_pose_score=0.15, min_part_score=0.1)
                    cv2.imshow("bruno", overlay_image)
                    cv2.waitKey(1)


                cv2.imshow("bruno", bruno)

                # compute score
                # write score and pause for a second

                cv2.waitKey(1000)




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
