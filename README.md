# Real-time Lane Detection and Tracking
## Overview
This project is an implementation of a real-time lane detection and tracking system using the OpenCV library. The system is designed to identify and track lane lines in a video stream or from a live camera feed. The main goal of the project is to develop an efficient, reliable, and accurate lane detection and tracking system that can be used in various applications such as Advanced Driver Assistance Systems (ADAS) and autonomous vehicles.

## Dependencies
1. Python 3.x
2. OpenCV (cv2)
3. NumPy

## How to Run
1. Install the required dependencies.
2. Save the code in a file named 'FinalVersion_LaneDetection.py'.
3. Run the script using the command python 'FinalVersion_LaneDetection.py'.

## Code Description
The code performs the following steps:

1. Define the region of interest and perspective transform parameters.
2. Define the sliding window parameters for lane line detection.
3. Read in the input video.
4. Apply perspective transform and thresholding to obtain a binary image.
5. Use the sliding windows technique to detect the left and right lane lines.
6. Fit a second-order polynomial to each lane line.
7. Calculate the average curvature of the two lane lines.
8. Calculate the distance from the center of the lane.
9. Apply inverse perspective transform to obtain the lane region on the original image.
10. Display the lane curvature and distance from the center on the image.
11. Show the final result.
When running the script, the output video will be displayed in a window named "Lane Detection". Press the 'a' key to stop the video and close the window.

## Note
The current implementation is designed to work with the provided project1/project_video.mp4 video file. To use the code with a different video or a live camera feed, you may need to adjust the perspective transform parameters, sliding window parameters, and other constants in the code.

For live camera feed, replace the line:

#### cap = cv2.VideoCapture('project1/project_video.mp4')
with:

#### cap = cv2.VideoCapture(0)
And make sure your camera is connected and working properly.

## Acknowledgments
This project has been developed with reference to the work of Neel Dani. The original implementation and ideas have been adapted and modified for the purpose of this project.

## License
This project is licensed under the MIT License as provided by Neel Dani. The original license is reproduced below for clarity:

MIT License

Copyright (c) 2019 Neel Dani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
