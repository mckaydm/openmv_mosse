# This work is licensed under the MIT license.
# Copyright (c) 2013-2023 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE
#
# Hello World Example
#
# Welcome to the OpenMV IDE! Click on the green run arrow button below to run the script!

import sensor
import time
import mosse

sensor.reset()  # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)  # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)  # Set frame size to QVGA (320x240)
sensor.skip_frames(time=2000)  # Wait for settings take effect.
clock = time.clock()  # Create a clock object to track the FPS.

tracker = mosse.MOSSE()
bbox = [100, 70, 50, 50]  # Example bounding box [x, y, w, h]
# Note: bounding box is scaled up to next highest power of 2 internally.
# For w=50, h=50, the tracker will use 64x64 internally.
# This is because the FFT implementation is radix-2 only.

init_img = sensor.snapshot()

if tracker.start(init_img, bbox):
    print("tracker initialized")
    print("Ai")
    print(tracker.Ai)
    print(" ")
    print("Bi")
    print(tracker.Bi)
