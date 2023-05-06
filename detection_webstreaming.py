# import the necessary packages
from pyimagesearch.motion_detection.singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import requests

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
vs = VideoStream(usePiCamera=1).start()
#vs = VideoStream(src=0, usePiCamera=False).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock
	# initialize the motion detector and the total number of frames
	# read thus far
	md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	# initialize the object detection model
	base_options = core.BaseOptions(file_name = '/home/pi/examples/lite/examples/object_detection/'
	'raspberry_pi/efficientdet_lite0.tflite', use_coral = False, num_threads = 4)
	detection_options = processor.DetectionOptions(max_results = 3, score_threshold = 0.3)
	options = vision.ObjectDetectorOptions(base_options = base_options, 
	detection_options = detection_options)
	detector = vision.ObjectDetector.create_from_options(options)

	# initialize the time
	start_time = time.time()
	# initialize a nested list to store 5 consecutive object detection results
	all_category_names = [[]]*5
	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		image = cv2.flip(frame, -1)
		rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		input_tensor = vision.TensorImage.create_from_array(rgb_image)
		detection_result = detector.detect(input_tensor)
		category_names = [detection.categories[0].category_name for detection in 
		detection_result.detections if detection.categories[0].score >= 0.3]
		# store the past 5 object detection results
		all_category_names[int(time.time()) % 5] = category_names
		#print(all_category_names)
		# flatten the nested list to check if "stop sign" is detected in the past 5 frames
		all_names = [item for name_list in all_category_names for item in name_list]
		print(all_names)
		# send push notifications every 300 seconds if stop sign is not detected in the past
		# 5 frames
		if time.time() - start_time > 5 and int(time.time() - start_time) % 300 == 0 \
		and "stop sign" not in all_names:
			print("Your garage door has been left open for 5 minute.")	
			requests.post('https://maker.ifttt.com/trigger/Garage_door_open/with/key/'
			'bJQ4_kNMs9zEVNSSft4BAA')
		frame = imutils.resize(frame, width=400)
		frame = cv2.flip(frame, -1)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)
		# grab the current timestamp and draw it on the frame
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
		if total > frameCount:
			# detect motion in the image
			motion = md.detect(gray)
			# check to see if motion was found in the frame
			if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(frame, (minX, minY), (maxX, maxY),
					(0, 0, 255), 2)
		
		# update the background model and increment the total number
		# of frames read thus far
		md.update(gray)
		total += 1
		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()
		time.sleep(1)
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
