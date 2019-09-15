#!/usr/bin/env python3

import sys
import math
import time
import os
from io import BytesIO
from websocket_server import WebsocketServer
import string
import tensorflow as tf
import cv2
import argparse
import posenet
import base64
import numpy as np
from PIL import Image
import io

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

# Called for every client connecting (after handshake)
def new_client(client, server):
        print("New client connected and was given id %d" % client['id'])
        #server.send_message_to_all("Hey all, a new client has joined us")


# Called for every client disconnecting
def client_left(client, server):
        print("Client(%d) disconnected" % client['id'])


# Called when a client sends a message
def message_received(client, server, message):
        print("Client(%d) said: %s" % (client['id'], message))
        png_recovered = base64.decodebytes(base64.b64decode(message))
        imgdata = base64.b64decode(message)
        image = Image.open(io.BytesIO(imgdata))

        img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        #img = np.frombuffer(png_recovered, dtype="uint8")

        # img is what you decode, a numpy array
        scale_factor=0.7125
        input_image, draw_image, output_scale = posenet.process_input(img, scale_factor, output_stride)
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(model_outputs,feed_dict={'image:0': input_image})
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)
        keypoint_coords *= output_scale
        result = keypoint_coords
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                result = keypoint_coords[:pi]
        print(result[0])        
        server.send_message_to_all(result)

PORT=5000
HOST="0.0.0.0"
with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        server = WebsocketServer(PORT, HOST)
        server.set_fn_new_client(new_client)
        server.set_fn_client_left(client_left)
        server.set_fn_message_received(message_received)
        server.run_forever()

