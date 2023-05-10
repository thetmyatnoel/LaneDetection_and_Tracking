import cv2
import numpy as np

# Define the region of interest and perspective transform parameters
src_points = np.float32([[580, 460], [205, 720], [1110, 720], [703, 460]])
dst_points = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Define the sliding window parameters
n_windows = 9
window_margin = 150
min_pixels = 50

# Read in the input video
cap = cv2.VideoCapture('project1/project_video.mp4')

while True:
    # Read in the next frame from the video
    ret, frame = cap.read()

    if not ret:
        print("none")
        break

    # Apply perspective transform and thresholding to obtain a binary image
    warped = cv2.warpPerspective(frame, M, (1280, 720))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Create a black image of the same size as the input frame
    out_img = np.zeros_like(frame)

    # Find the starting points of the left and right lane lines
    histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set up the sliding windows
    window_height = int(binary.shape[0] / n_windows)
    nonzero = binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    # Loop through the windows and collect the non-zero pixels
    for window in range(n_windows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - window_margin
        win_xleft_high = leftx_current + window_margin
        win_xright_low = rightx_current - window_margin
        win_xright_high = rightx_current + window_margin

        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > min_pixels:
            leftx_current = int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > min_pixels:
            rightx_current = int(np.mean(nonzero_x[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    # Fit a second-order polynomial to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate the x and y values for plotting
    ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Draw the detected lane region
    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]
    cv2.polylines(out_img, [np.int32(np.vstack([left_fitx, ploty]).T)], False, (255, 255, 0), thickness=2)
    cv2.polylines(out_img, [np.int32(np.vstack([right_fitx, ploty]).T)], False, (255, 255, 0), thickness=2)

    # Calculate the average curvature of the two lane lines
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])
    avg_curverad = (left_curverad + right_curverad) / 2

    # Calculate the distance from the center of the lane
    center_offset_pix = (left_fitx[-1] + right_fitx[-1]) / 2 - binary.shape[1] / 2
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    center_offset_mtrs = center_offset_pix * xm_per_pix

    # Apply inverse perspective transform to obtain the lane region on the original image
    lane_region = cv2.addWeighted(frame, 1, cv2.warpPerspective(out_img, np.linalg.inv(M), (1280, 720)), 0.3, 0)

    # Draw the lane curvature on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Lane Curvature: {:.2f} m'.format(avg_curverad)
    cv2.putText(lane_region, text, (50, 50), font, 1, (255, 255, 255), 2)

    # Draw the distance from center on the image
    text = 'Vehicle is {:.2f} m {} of center'.format(abs(center_offset_mtrs), 'left' if center_offset_mtrs <= 0 else 'right')
    cv2.putText(lane_region, text, (50, 100), font, 1, (255, 255, 255), 2)

    # Show the final result
    cv2.imshow('Lane Detection', lane_region)

    if cv2.waitKey(1) == ord('a'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()


