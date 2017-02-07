import numpy as np
import glob
import math
import matplotlib.image as mpimg
import cv2

from lane_assist_pipeline_ops import LaneAssistOp, CameraCalibrationOp, PlotImageOp

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

# This constant ultimately contributes to deriving a given
# period when computing SMA and EMA for line noise smoothing
FPS = 30

class LaneLine:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def angle(self):
        return math.atan2(self.y2 - self.y1, self.x2 - self.x1) * 180.0 / np.pi

    def slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def y_intercept(self):
        return self.y1 - self.slope() * self.x1

    def __str__(self):
        return "(x1, y1, x2, y2, slope, y_intercept, angle) == (%s, %s, %s, %s, %s, %s, %s)" % (
            self.x1, self.y1, self.x2, self.y2, self.slope(), self.y_intercept(), self.angle())


class PipelineRunner:
    def __init__(self, calibration_op, ema_period_alpha=0.65):
        self.current_frame = 0
        self.__calibration_op = calibration_op

        self.l_poly_coefficients = np.array([[],[],[]])
        self.l_ema = np.array([0,0,0])

        self.r_poly_coefficients = np.array([[],[],[]])
        self.r_ema = np.array([0,0,0])

        self.ema_fps_period = ema_period_alpha * FPS

    def process_video(self, src_video_path, dst_video_path, audio=False):
        self.current_frame = 0
        VideoFileClip(src_video_path).fl_image(self.process_image).write_videofile(dst_video_path, audio=audio)

    def process_image(self, image):
        self.current_frame += 1
        cv2.imwrite('processed_images/frame_{}_in.jpg'.format(self.current_frame), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        result = LaneAssistOp(
            image,
            self.__calibration_op,
            margin=100,
            kernel_size=15,
            sobelx_thresh=(20,100),
            sobely_thresh=(20,100),
            mag_grad_thresh=(20,250),
            dir_grad_thresh=(0.3, 1.3)
        ).perform().output()

        cv2.imwrite('processed_images/frame_{}_out.jpg'.format(self.current_frame), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        # PlotImageOp(result, title="{} - frame {}".format(self.__class__.__name__, current_frame)).perform()
        return result

    def draw_lane(self, undistorted, binary_warped, fit_leftx, fit_rightx, fity, warper_op):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 195, 255))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, warper_op['Minv'], (binary_warped.shape[1], binary_warped.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.5, 0)
        cv2.imwrite('detected_lanes/frame_'+self.current_frame+'.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    def compute_ema(self, measurement, all_measurements, curr_ema):
        sma = sum(all_measurements) / (len(all_measurements))

        if len(all_measurements) < self.ema_fps_period:
            # let's just use SMA until
            # our EMA buffer is filled
            return sma

        multiplier = 2 / float(len(all_measurements) + 1)
        ema = (measurement - curr_ema) * multiplier + curr_ema

        # print("sma: %s, multiplier: %s" % (sma, multiplier))
        return ema

    @staticmethod
    def compute_least_squares_line(lines):
        all_x1 = []
        all_y1 = []
        all_x2 = []
        all_y2 = []

        for line in lines:
            x1, y1, x2, y2, angle, m, b = line.x1, line.y1, line.x2, line.y2, line.angle(), line.slope(), line.y_intercept()
            all_x1.append(x1)
            all_y1.append(y1)
            all_x2.append(x2)
            all_y2.append(y2)

        all_x = (all_x1 + all_x2)
        all_y = (all_y1 + all_y2)

        n = len(all_x)

        all_x_y_dot_prod = sum([xi * yi for xi, yi in zip(all_x, all_y)])
        all_x_squares = sum([xi ** 2 for xi in all_x])

        a = ((n * all_x_y_dot_prod) - (sum(all_x) * sum(all_y))) / ((n * all_x_squares) - (sum(all_x) ** 2))
        b = ((sum(all_y) * all_x_squares) - (sum(all_x) * all_x_y_dot_prod)) / ((n * all_x_squares) - (sum(all_x) ** 2))

        return a, b

if __name__ == '__main__':
    calibration_images = glob.glob('camera_cal/calibration*.jpg')
    calibration_op = CameraCalibrationOp(calibration_images=calibration_images, x_inside_corners=9, y_inside_corners=6).perform()

    pipeline_runner = PipelineRunner(calibration_op)

    # result = pipeline_runner.process_image(mpimg.imread('test_images/frame_9_in.jpg'))
    pipeline_runner.process_video('project_video.mp4', 'project_video_final.mp4')