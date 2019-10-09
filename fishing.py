#############################################
# Import Libraries and modules
#############################################
import logging
import random
import numpy as np
from collections import deque
import pyscreenshot
import cv2
import pyautogui
import pyaudio
import audioop
import time
from av_helpers.file_helpers import ensure_dir

#############################################
# Warning Filter
#############################################


#############################################
# Logging for Module
#############################################
logger_name = "fishbot_logger"
FORMAT = 'apimethod''%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)

#############################################
# Global Variables
#############################################
random.seed(time.time())


#############################################
# Function / Class Definitions
#############################################
class FishBot(object):
    def __init__(self, img_thresh=0.8, sound_thresh=200, box=(0.3, 0.2, 0.7, 0.8)):
        self.img_thresh = img_thresh
        self.sound_thresh = sound_thresh
        self.screen_size = pyscreenshot.grab().size
        self.save_success = False
        self.box_start_point = (self.screen_size[0] * box[0], self.screen_size[1] * box[1])
        self.box_end_point = (self.screen_size[0] * box[2], self.screen_size[1] * box[3])
        logger.info(f"Current screenshot box: top-left{self.box_start_point}, "
                    f"bottom-right {self.box_end_point}, CORR threshold {self.img_thresh}")
        self.background_rms = self.get_background_sound_rms_benchmark()
        logger.info(f"Current background sound RMS: {self.background_rms}, "
                    f"RMS_threshold: {sound_thresh}")

    @staticmethod
    def logout():
        pyautogui.typewrite(message="/logout\n", interval=0.2)

    @staticmethod
    def login():
        pyautogui.press(keys='enter')
        time.sleep(30 + random.random() * 10)  # wait for loading

    @staticmethod
    def send_float():
        logger.info("Sending float")
        pyautogui.press(keys='1')
        logger.info("Wait for animation")
        time.sleep(2)

    @staticmethod
    def move_mouse(cursor_xy):
        logger.info("Moving cursor to " + str(cursor_xy))
        pyautogui.moveTo(x=cursor_xy[0], y=cursor_xy[1],
                         duration=0.3 + 0.3 * random.random(),
                         tween=pyautogui.easeOutBounce)

    def reset_mouse(self):
        pyautogui.moveTo(x=self.box_end_point[0], y=self.box_end_point[1],
                         duration=0.3 + 0.3 * random.random(),
                         tween=pyautogui.easeOutCirc)

    @staticmethod
    def snatch():
        logger.info('Snatching!')
        time.sleep(random.random())
        pyautogui.click(button='right', clicks=2, interval=0.4, duration=0.3)

    def make_screenshot(self, i=None):
        logger.info("Capturing screen")
        screenshot = pyscreenshot.grab(bbox=(self.box_start_point[0], self.box_start_point[1],
                                             self.box_end_point[0], self.box_end_point[1]))
        if i is None:
            screenshot_filepath = ensure_dir('fishing_session/screenshot.png')
        else:
            screenshot_filepath = ensure_dir(f'fishing_session/fishing_float_{i}.png')
        screenshot.save(screenshot_filepath)

    def find_float(self):
        logger.info("searching for float")
        for i in range(0, 4):
            template = cv2.imread(f'float_template/fishing_float_{i}.png', 1)
            screenshot = cv2.imread('fishing_session/screenshot.png', 1)
            w, h = template.shape[1::-1]
            result = cv2.matchTemplate(image=screenshot, templ=template, method=cv2.TM_CCORR_NORMED)
            corr_min, corr_max, min_loc, max_loc = cv2.minMaxLoc(result)
            tagged_img = cv2.rectangle(img=screenshot, pt1=max_loc,
                                       pt2=(max_loc[0] + w, max_loc[1] + h),
                                       color=(0, 0, 255), thickness=2, )
            timestamp = time.strftime('%m%d%H%M%S', time.localtime())
            if corr_max >= self.img_thresh:
                logger.info(f"Found float {i}!")
                if self.save_success:
                    filename = ensure_dir(f'fishing_session/success/{corr_max:.2f}_{timestamp}.png')
                    cv2.imwrite(filename=filename, img=tagged_img)
                float_coordinate_in_box = (max_loc[0] + 2 * w / 3, max_loc[1] + 2 * h / 3)
                float_cursor_x = self.box_start_point[0] + float_coordinate_in_box[0]
                float_cursor_y = self.box_start_point[1] + float_coordinate_in_box[1]
                return int(float_cursor_x), int(float_cursor_y)
            else:
                logger.info(f"Failed to find the float. Save the screenshot for analysis.")
                filename = ensure_dir(f'fishing_session/failed/{corr_max:.2f}_{timestamp}.png')
                cv2.imwrite(filename=filename, img=tagged_img)
                return None

    @staticmethod
    def get_background_sound_rms_benchmark():
        logger.info("Checking for the benchmark RMS of background sounds...")
        background_test_duration = 5

        p = pyaudio.PyAudio()
        input_device_info = p.get_default_input_device_info()
        channels = int(input_device_info['maxInputChannels'])
        rate = int(input_device_info['defaultSampleRate'])
        chunk = 1024  # number of frames to read each time
        stream = p.open(rate=rate,
                        channels=channels,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=chunk)
        rms_window = list()

        listening_start_time = time.time()
        while time.time() - listening_start_time <= background_test_duration:
            try:
                data = stream.read(chunk)
                rms_window.append(audioop.rms(data, 2))
            except IOError:
                break
        stream.stop_stream()
        stream.close()
        p.terminate()

        return np.mean(rms_window)

    def listen_splash(self):
        logger.info("listening for splash sounds...")
        splash_sound_duration = 0.2
        fishing_duration = 30

        # Open stream
        p = pyaudio.PyAudio()
        input_device_info = p.get_default_input_device_info()
        channels = int(input_device_info['maxInputChannels'])
        rate = int(input_device_info['defaultSampleRate'])
        chunk = 1024  # number of frames to read each time
        stream = p.open(rate=rate,
                        channels=channels,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=chunk)
        rms_window = deque(maxlen=int((splash_sound_duration * rate) / chunk))

        success = False
        listening_start_time = time.time()
        while True:
            try:
                data = stream.read(chunk)
                rms_window.append(audioop.rms(data, 2))
                rms = np.mean(rms_window)
                if rms > self.sound_thresh:
                    logger.info(f"RMS: {rms}. Heard something loud!")
                    success = True
                    break
                if time.time() - listening_start_time > fishing_duration:
                    logger.info("Failed to hear anything in 20 seconds!")
                    break
            except IOError:
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

        return success

    def get_fishing_float_template(self, n=20):
        for i in range(n):
            self.send_float()
            self.make_screenshot(i)

    def start_fish(self):
        tries = 0
        catched = 0

        while True:
            self.reset_mouse()
            tries += 1
            self.send_float()
            self.make_screenshot()
            cursor_xy = self.find_float()
            if cursor_xy is None:
                logger.info("Float was not found, retrying in 1 seconds")
                time.sleep(random.random())
                continue
            logger.info(f"Float found at {cursor_xy}")
            self.move_mouse(cursor_xy)
            if not self.listen_splash():
                time.sleep(1)
                continue
            self.snatch()
            logger.info("snatched something!")
            catched += 1
            time.sleep(random.random() * 3)
            if catched == 500:
                break


#############################################
# Main Function
#############################################
if __name__ == '__main__':
    fb = FishBot(img_thresh=0.8, sound_thresh=180, box=(0.3, 0.2, 0.7, 0.8))
    fb.start_fish()
    # fb.get_fishing_float_template(n=20)
