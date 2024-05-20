from lane import Lane

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from timeit import default_timer as timer


class LaneDetector(object):
    def __init__(self):
        self.left_lane = Lane()
        self.right_lane = Lane()
        self.frame_width = 1280
        self.frame_height = 720

        self.LOCALIZATION = "RU"

        self.RU_DICT = {
            "Left curve": "левее",
            "Right curve": "правее",
            "Straight": "прямо",
            "Left": "влево",
            "Right": "вправо",
        }

        self.LANEWIDTH = 3.75  # highway lane width: 3.75 meters
        self.THRESHOLD = 0.5
        self.INITIAL_OFFSET = -0.2 # set 0.0 if camera is in center of car
        self.input_scale = 2
        self.output_frame_scale = 1

        # fullsize:1280x720
        self.x = [194, 1117, 685, 535]
        self.y = [719, 719, 461, 461]
        self.X = [200, 1200, 1200, 200]
        self.Y = [719, 719, 0, 0]

        '''
        self.x = [194, 1117, 685, 535]
        self.y = [719, 719, 461, 461]
        self.X = [200, 1200, 1200, 200]
        self.Y = [719, 719, 0, 0]
        '''
        self.src = np.floor(np.float32([[self.x[0], self.y[0]], [self.x[1], self.y[1]],[self.x[2], self.y[2]], [self.x[3], self.y[3]]]) / self.input_scale)
        self.dst = np.floor(np.float32([[self.X[0], self.Y[0]], [self.X[1], self.Y[1]],[self.X[2], self.Y[2]], [self.X[3], self.Y[3]]]) / self.input_scale)

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

        # Threshold for color and gradient thresholding
        self.s_thresh, self.sx_thresh, self.dir_thresh, self.m_thresh, self.r_thresh = (120, 255), (20, 100), (0.7, 1.3), (30, 100), (200, 255)

        # Number of sliding windows
        self.nwindows = 5

        # Approx flag (based on curvature)
        self.approx_flag = 0

    @staticmethod
    def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        # 3) Take the absolute value of the derivative or gradient
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255. * abs_sobel / np.max(abs_sobel))

        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output

    @staticmethod
    def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Calculate the magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        # 5) Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        return binary_output

    @staticmethod
    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the x and y gradients
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        return binary_output

    @staticmethod
    def threshold_col_channel(channel, thresh):
        binary = np.zeros_like(channel)
        binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

        return binary

    def find_edges(self, img):
        img = np.copy(img)
        # Convert to HSV color space and threshold the s channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        s_channel = hls[:, :, 2]
        s_binary = self.threshold_col_channel(s_channel, thresh=self.s_thresh)

        # Sobel x
        sxbinary = self.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=self.sx_thresh)
        # mag_binary = mag_thresh(img, sobel_kernel=3, thresh=m_thresh)
        # # gradient direction
        dir_binary = self.dir_threshold(img, sobel_kernel=3, thresh=self.dir_thresh)
        #
        # # output mask
        combined_binary = np.zeros_like(s_channel)
        combined_binary[(((sxbinary == 1) & (dir_binary == 1)) | ((s_binary == 1) & (dir_binary == 1)))] = 1

        # add more weights for the s channel
        c_bi = np.zeros_like(s_channel)
        c_bi[((sxbinary == 1) & (s_binary == 1))] = 2

        ave_binary = (combined_binary + c_bi)

        return ave_binary

    @staticmethod
    def warper(img, M):
        # Compute and apply perspective transform
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped

    def full_search(self, binary_warped, visualization=False):
        # fit the lane line
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        out_img = out_img.astype('uint8')

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = np.floor(100 / self.input_scale)
        # Set minimum number of pixels found to recenter window
        minpix = np.floor(50 / self.input_scale)
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            if visualization:
                cv2.rectangle(out_img, (int(win_xleft_low), int(win_y_low)), (int(win_xleft_high), int(win_y_high)),
                              (0, 255, 0), 2)
                cv2.rectangle(out_img, (int(win_xright_low), int(win_y_low)), (int(win_xright_high), int(win_y_high)),
                              (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Visualization

        # Generate x and y values for plotting
        if visualization:
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            # plt.subplot(1,2,1)
            plt.imshow(out_img)
            # plt.imshow(binary_warped)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim((0, self.frame_width / self.input_scale))
            plt.ylim((self.frame_height / self.input_scale, 0))
            plt.show()

        return left_fit, right_fit

    def window_search(self, left_fit, right_fit, binary_warped, margin=100, visualization=False):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's easier to find line pixels with windows search
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
                    (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
                    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if visualization:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            # And you're done! But let's visualize the result here as well
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            out_img = out_img.astype('uint8')
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            plt.imshow(result)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim((0, self.frame_width / self.input_scale))
            plt.ylim((self.frame_height / self.input_scale, 0))

            plt.show()

        return left_fit, right_fit

    def measure_lane_curvature(self, ploty, leftx, rightx, visualization=False):

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / (self.frame_height / self.input_scale)  # meters per pixel in y dimension
        xm_per_pix = self.LANEWIDTH / (700 / self.input_scale)  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')

        if leftx[0] - leftx[-1] > 100 / self.input_scale:
            curve_direction = 'Left curve'
        elif leftx[-1] - leftx[0] > 100 / self.input_scale:
            curve_direction = 'Right curve'
        else:
            curve_direction = 'Straight'

        return (left_curverad + right_curverad) / 2.0, curve_direction

    def off_center(self, left, mid, right):
        a = mid - left
        b = right - mid
        width = right - left

        if a >= b:  # driving right off
            offset = a / width * self.LANEWIDTH - self.LANEWIDTH / 2.0 + self.INITIAL_OFFSET
        else:  # driving left off
            offset =self.LANEWIDTH / 2.0 - b / width * self.LANEWIDTH + self.INITIAL_OFFSET

        return offset

    def compute_car_offcenter(self, ploty, left_fitx, right_fitx, undist):

        # Create an image to draw the lines on
        height = undist.shape[0]
        width = undist.shape[1]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        bottom_l = left_fitx[height - 1]
        bottom_r = right_fitx[0]

        offcenter = self.off_center(bottom_l, width / 2.0, bottom_r)

        return offcenter, pts

    def tracker(self, binary_sub, ploty, visualization=False):
        left_fit, right_fit = self.window_search(self.left_lane.prev_poly, self.right_lane.prev_poly, binary_sub,
                                                 margin=int(100 / self.input_scale), visualization=visualization)

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        std_value = np.std(right_fitx - left_fitx)
        if std_value < (85 / self.input_scale):
            self.left_lane.detected = True
            self.right_lane.detected = True
            self.left_lane.current_poly = left_fit
            self.right_lane.current_poly = right_fit
            self.left_lane.cur_fitx = left_fitx
            self.right_lane.cur_fitx = right_fitx
            # global tt
            # tt = tt + 1
        else:
            self.left_lane.detected = False
            self.right_lane.detected = False
            self.left_lane.current_poly = self.left_lane.prev_poly
            self.right_lane.current_poly = self.right_lane.prev_poly
            self.left_lane.cur_fitx = self.left_lane.prev_fitx[-1]
            self.right_lane.cur_fitx = self.right_lane.prev_fitx[-1]

    def detector(self, binary_sub, ploty, visualization=False):
        left_fit, right_fit = self.full_search(binary_sub, visualization=visualization)

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        std_value = np.std(right_fitx - left_fitx)
        if std_value < (85 / self.input_scale):
            self.left_lane.current_poly = left_fit
            self.right_lane.current_poly = right_fit
            self.left_lane.cur_fitx = left_fitx
            self.right_lane.cur_fitx = right_fitx
            self.left_lane.detected = True
            self.right_lane.detected = True
        else:
            self.left_lane.current_poly = self.left_lane.prev_poly
            self.right_lane.current_poly = self.right_lane.prev_poly
            if len(self.left_lane.prev_fitx) > 0:
                self.left_lane.cur_fitx = self.left_lane.prev_fitx[-1]
                self.right_lane.cur_fitx = self.right_lane.prev_fitx[-1]
            else:
                self.left_lane.cur_fitx = left_fitx
                self.right_lane.cur_fitx = right_fitx
            self.left_lane.detected = False
            self.right_lane.detected = False

    def create_output_frame(self, offcenter, pts, img_ori, fps, curvature, curve_direction, binary_sub):
        img_ori = cv2.resize(img_ori, (0, 0), fx=1 / self.output_frame_scale, fy=1 / self.output_frame_scale)
        w = img_ori.shape[1]
        h = img_ori.shape[0]

        color_warp = np.zeros_like(img_ori).astype(np.uint8)

        # create a frame to hold every image
        whole_frame = np.zeros((int(h * 2.5), int(w * 2.34), int(3)), dtype=np.uint8)

        if abs(offcenter) > self.THRESHOLD:  # car is offcenter more than 0.6 m
            # Draw Red lane
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))  # red
        else:  # Draw Green lane
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))  # green

        newwarp = cv2.warpPerspective(color_warp, self.M_inv,
                                      (int(self.frame_width / self.input_scale), int(self.frame_height / self.input_scale)))

        # Combine the result with the original image    # result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        newwarp_ = cv2.resize(newwarp, None, fx=self.input_scale / self.output_frame_scale, fy=self.input_scale / self.output_frame_scale,
                              interpolation=cv2.INTER_LINEAR)

        output = cv2.addWeighted(img_ori, 1, newwarp_, 0.3, 0)

        if offcenter >= 0:
            offset = offcenter
            direction = 'Right'
        elif offcenter < 0:
            offset = -offcenter
            direction = 'Left'

        font = cv2.FONT_HERSHEY_COMPLEX

        info_framerate = "{0:4.1f} FPS".format(fps)
        if self.LOCALIZATION == "RU":
            info_title = "Контроль полосы и дистанции"
            info_lane = "Направление: {0}".format(self.RU_DICT[curve_direction])
            info_cur = "Кривизна {:6.1f} м".format(curvature)
            info_offset = "Смещение: {0} {1:3.1f} м".format(self.RU_DICT[direction], offset)
        else:
            info_title = "Lane Departure and Distance Control"
            info_lane = "Lane: {0}".format(curve_direction)
            info_cur = "Curvature {:6.1f} m".format(curvature)
            info_offset = "Off center: {0} {1:3.1f}m".format(direction, offset)

        cv2.rectangle(output, (0, 0), (400, 175), color=0, thickness=-1)
        cv2.rectangle(output, (0, 0), (1280, 50), color=0, thickness=-1)
        cv2.putText(output, info_title, (20, 35), font, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(output, info_framerate, (20, 60), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(output, info_lane, (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(output, info_cur, (20, 125), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        if abs(offcenter) > self.THRESHOLD:
            cv2.putText(output, info_offset, (20, 150), font, 0.8, (0, 0, 255), 1,
                        cv2.LINE_AA)
        else:
            cv2.putText(output, info_offset, (20, 155), font, 0.8, (0, 255, 0), 1,
                        cv2.LINE_AA)

        return output

    @staticmethod
    def approx_lane_ahead(img, vehicle_ahead):
        left = vehicle_ahead['box'][0]
        top = vehicle_ahead['box'][1]
        width = vehicle_ahead['box'][2]
        height = vehicle_ahead['box'][3]

        # Left line
        bottom_left = [left - 30, top + height]
        length = 600
        angle = 180-40
        left_line_coord =(
            round(bottom_left[0] + length * math.cos(angle * math.pi / 180.0)),
            round(bottom_left[1] + length * math.sin(angle * math.pi / 180.0))
        )
        img = cv2.line(img, bottom_left, left_line_coord, (255, 255, 255), 5)

        length = 100
        angle = 180-40
        left_line_coord = (
            round(bottom_left[0] - length * math.cos(angle * math.pi / 180.0)),
            round(bottom_left[1] - length * math.sin(angle * math.pi / 180.0))
        )
        img = cv2.line(img, bottom_left, left_line_coord, (255, 255, 255), 5)

        # Right line
        bottom_right = [left + width + 30, top + height]
        length = 600
        angle = 40
        right_line_coord = (
            round(bottom_right[0] + length * math.cos(angle * math.pi / 180.0)),
            round(bottom_right[1] + length * math.sin(angle * math.pi / 180.0))
        )
        img = cv2.line(img, bottom_right, right_line_coord, (255, 255, 255), 5)

        length = 100
        angle = 40
        right_line_coord = (
            round(bottom_right[0] - length * math.cos(angle * math.pi / 180.0)),
            round(bottom_right[1] - length * math.sin(angle * math.pi / 180.0))
        )
        img = cv2.line(img, bottom_right, right_line_coord, (255, 255, 255), 5)
        return img

    def process_frame(self, img, vehicle_ahead, visualization=False):
        start = timer()

        if self.approx_flag:
            try:
                self.left_lane.detected = False
                self.right_lane.detected = False
                img = self.approx_lane_ahead(img, vehicle_ahead)
            except:
                pass
        # resize the input image according to scale
        img_resized = cv2.resize(img, (0, 0), fx=1 / self.input_scale, fy=1 / self.input_scale)

        # find the binary image of lane/edges
        img_binary = self.find_edges(img_resized)

        # warp the image to bird view
        binary_warped = self.warper(img_binary, self.M)  # get binary image contains edges

        # crop the binary image
        binary_sub = np.zeros_like(binary_warped)
        binary_sub[:, int(150 / self.input_scale):int(-80 / self.input_scale)] = \
            binary_warped[:, int(150 / self.input_scale):int(-80 / self.input_scale)]

        # start detector or tracker to find the lanes
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        if self.left_lane.detected:  # start tracker
            self.tracker(binary_sub, ploty, visualization)
        else:  # start detector
            self.detector(binary_sub, ploty, visualization)

        # average among the previous N frames to get the averaged lanes
        self.left_lane.process(ploty)
        self.right_lane.process(ploty)

        # measure the lane curvature
        curvature, curve_direction = self.measure_lane_curvature(ploty, self.left_lane.mean_fitx, self.right_lane.mean_fitx)

        # Check approx flag criterion
        if curvature < 500:
            self.approx_flag = 1
        elif curvature > 5000:
            self.approx_flag = 0

        # compute the car's off-center in meters
        offcenter, pts = self.compute_car_offcenter(ploty, self.left_lane.mean_fitx, self.right_lane.mean_fitx, img_resized)

        # compute the processing frame rate
        end = timer()
        fps = 1.0 / (end - start)

        # combine all images into final video output (only for visualization purpose)
        output = self.create_output_frame(offcenter, pts, img, fps, curvature, curve_direction, binary_sub)
        return output
