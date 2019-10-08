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
import wave
import audioop
import time
import math

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
	def __init__(self, threshold=0.8, box=(0.3, 0.3, 0.7, 0.7)):
		self.threshold = threshold
		img = pyscreenshot.grab()
		self.screen_size = img.size
		self.box_start_point = (self.screen_size[0] * box[0], self.screen_size[1] * box[1])
		self.box_end_point = (self.screen_size[0] * box[2], self.screen_size[1] * box[3])

	@staticmethod
	def logout():
		pyautogui.typewrite(message="/logout\n", interval=0.2)

	@staticmethod
	def login():
		pyautogui.press(keys='enter')
		time.sleep(30 + random.random()*10)  # wait for loading

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
						 duration=0.3+ 0.3*random.random(),
						 tween=pyautogui.easeOutBounce)

	@staticmethod
	def snatch():
		logger.info('Snatching!')
		pyautogui.click(button='right')

	def make_screenshot(self):
		logger.info("Capturing screen")
		screenshot = pyscreenshot.grab(bbox=(self.box_start_point[0], self.box_start_point[1],
											 self.box_end_point[0], self.box_end_point[1]))
		screenshot_filepath = 'fishing_session/screenshot.png'
		screenshot.save(screenshot_filepath)

	def find_float(self):
		logger.info("searching for float")
		for i in range(0, 7):
			template = cv2.imread(f'float_template/fishing_float_{i}.png', 0)
			screenshot = cv2.imread('fishing_session/screenshot.png')
			w, h = template.shape[1::-1]
			result = cv2.matchTemplate(image=screenshot, templ=template, method=cv2.TM_CCORR_NORMED)
			corr_min, corr_max, min_loc, max_loc = cv2.minMaxLoc(result)
			tagged_img = cv2.rectangle(img=screenshot, pt1=max_loc,
										   pt2=(max_loc[0] + w, max_loc[1] + h),
										   color=(0,0,255), thickness=2, )
			timestamp = time.strftime('%m%d%H%M%S', time.localtime())
			if corr_max >= self.threshold:
				logger.info(f"Found float {i}!")
				filename = f'fishing_session/success/corr_{corr_max}_{timestamp}.png'
				cv2.imwrite(filename=filename, img=tagged_img)
				float_coordinate_in_box = (max_loc[0] + 2* w / 3, max_loc[1] + 2* h / 3)
				float_cursor_x = self.box_start_point[0] + float_coordinate_in_box[0]
				float_cursor_y = self.box_start_point[1] + float_coordinate_in_box[1]
				return int(float_cursor_x), int(float_cursor_y)
			else:
				logger.info(f"Failed to find the float. Save the screenshot for analysis.")
				filename = f'fishing_session/failed/corr_{corr_max}_{timestamp}.png'
				cv2.imwrite(filename=filename, img=tagged_img)
				return None

	def listen_splash(self):
		logger.info("listening for loud splash sounds...")
		CHUNK = 1024  # CHUNKS of bytes to read each time
		FORMAT = pyaudio.paInt16
		CHANNELS = 2
		RATE = 18000
		THRESHOLD = 1200  # The threshold intensity that defines silence
						  # and noise signal (an int. lower than THRESHOLD is silence).
		SILENCE_LIMIT = 1  # Silence limit in seconds. The max ammount of seconds where
						   # only silence is recorded. When this time passes the
						   # recording finishes and the file is delivered.
		#Open stream
		p = pyaudio.PyAudio()

		stream = p.open(format=FORMAT,
						channels=CHANNELS,
						rate=RATE,
						input=True,
						frames_per_buffer=CHUNK)
		cur_data = ''  # current chunk  of audio data
		rel = RATE/CHUNK
		slid_win = deque(maxlen=SILENCE_LIMIT * rel)


		success = False
		listening_start_time = time.time()
		while True:
			try:
				cur_data = stream.read(CHUNK)
				slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
				if(sum([x > THRESHOLD for x in slid_win]) > 0):
					print 'I heart something!'
					success = True
					break
				if time.time() - listening_start_time > 20:
					print 'I don\'t hear anything already 20 seconds!'
					break
			except IOError:
				break

		# print "* Done recording: " + str(time.time() - start)
		stream.close()
		p.terminate()
		return success

def main():
	if check_process() and not dev:
		print "Waiting 2 seconds, so you can switch to WoW"
		time.sleep(2)

	check_screen_size()
	catched = 0
	tries = 0
	while not dev:
		tries += 1
		send_float()
		im = make_screenshot()
		place = find_float(im)
		if not place:
			print 'Float was not found, retrying in 2 seconds'
			time.sleep(3)
			im = make_screenshot()
			place = find_float(im)
			if not place:
				print 'Still can\'t find float, breaking this session'
				jump()
				continue
		print('Float found at ' + str(place))
		move_mouse(place)
		if not listen():
			print 'If we didn\' hear anything, lets try again'
			jump()
			continue
		snatch()
		time.sleep(3)
		catched += 1
		print 'I guess we\'ve snatched something'
		if catched == 250:
			break
		time.sleep(3)
	print 'catched ' + str(catched)
	logout()

main()
