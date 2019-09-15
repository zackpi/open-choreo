import json
import glob
import os
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

files = sorted(glob.glob("keypoints/*"))


PART_NAMES = [
  "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
  "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
  "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

poses = []
for file in files:
	with open(file) as f:
		raw = f.read()
		if raw:
			jsobj = json.loads(raw)
			poses.append(jsobj["Pose0"])

velx = []
vely = []
velsq = []
for t in range(1, len(poses)):
	then_pose, now_pose = poses[t-1], poses[t]

	dx = []
	for i in range(17):
		dx.append(now_pose[PART_NAMES[i]]["x"] - then_pose[PART_NAMES[i]]["x"])
	velx.append(-sum(dx) / 17)

xpeaks, _ = find_peaks(velx, prominence=2)

plt.plot(range(len(velx)), velx)
plt.plot(xpeaks, [velx[peak] for peak in xpeaks], "x")
plt.show()


segments = []
if len(xpeaks) == 0:
	segments = poses
else:

	segments.append(poses[:xpeaks[0]])
	for t in range(len(xpeaks)-1):
		segments.append(poses[xpeaks[t]:xpeaks[t+1]])
	segments.append(poses[xpeaks[-1]:])


with open("segments.json", 'w') as fp:
  json.dump(segments, fp)
