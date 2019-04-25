# Copyright (c) 2019 Wrnch Inc.
# All rights reserved

from __future__ import print_function, division

import cv2
import wrnchAI
import sys
from visualizer import Visualizer

if not wrnchAI.license_check():
    sys.exit('A valid license is required to run the samples')

num_args = len(sys.argv)
if num_args < 2 or num_args > 3:
    sys.exit(
        'Usage: python pose2d_sample.py <model path> [camera index {0}]')

if num_args == 3:
    webcam_index = int(sys.argv[2])
else:
    webcam_index = 0

params = wrnchAI.PoseParams()
params.bone_sensitivity = wrnchAI.Sensitivity.High
params.joint_sensitivity = wrnchAI.Sensitivity.High
params.enable_tracking = True

# Default Model resolution
params.preferred_net_width = 328
params.preferred_net_height = 184

output_format = wrnchAI.JointDefinitionRegistry.get('j23')

print('Initializing networks...')
estimator = wrnchAI.PoseEstimator(
    models_path=sys.argv[1], params=params, gpu_id=0, output_format=output_format)
print('Initialization done!')

options = wrnchAI.PoseEstimatorOptions()

print('Opening webcam...')
cap = cv2.VideoCapture(webcam_index)

if not cap.isOpened():
    sys.exit('Cannot open webcam.')

visualizer = Visualizer()

joint_definition = estimator.human_2d_output_format()
bone_pairs = joint_definition.bone_pairs()

while True:
    ret, frame = cap.read()

    if frame is not None:
        estimator.process_frame(frame, options)
        humans2d = estimator.humans_2d()

        visualizer.draw_image(frame)
        for human in humans2d:
            joints = human.joints()

            visualizer.draw_points(joints)
            visualizer.draw_lines(joints, bone_pairs)

        visualizer.show()

    key = cv2.waitKey(1)

    if key & 255 == 27:
        break

cap.release()
cv2.destroyAllWindows()
