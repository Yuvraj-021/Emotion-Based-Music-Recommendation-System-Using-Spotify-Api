import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
from keras.layers import Activation, Conv2D, LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
from keras.preprocessing import image
import datetime
from threading import Thread
import time
import recognition.Emotion as re
import pandas as pd


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials



face_cascade=cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
ds_factor=0.6

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model/model.h5')

cv2.ocl.setUseOpenCL(False)
predicted_emotion = ''

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv ",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text=[0]

# define a variable to hold the predicted emotion
predicted_emotion = ''


stop_camera_flag=False


def stop_camera():
    global stop_camera_flag
    stop_camera_flag = True
    WebcamVideoStream.stop
    # a=VideoCamera()
    # a.spotifyMusic()
    #cap1.release()


''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()


''' Class for using another thread for video streaming to boost performance '''
class WebcamVideoStream:
    	
		def __init__(self, src=0):
			self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
			(self.grabbed, self.frame) = self.stream.read()
			self.stopped = False

		def start(self):
				# start the thread to read frames from the video stream
			Thread(target=self.update, args=()).start()
			return self
			
		def update(self):
			#i=1;
			# keep looping infinitely until the thread is stopped
			while True:
				#i=i+1
				if stop_camera_flag:
					return
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					return
				# otherwise, read the next frame from the stream
				(self.grabbed, self.frame) = self.stream.read()

		def read(self):
			# return the frame most recently read
			return self.frame
		def stop(self):
			# indicate that the thread should be stopped
			self.stopped = True

''' Class for reading video stream, generating prediction and recommendations '''
class VideoCamera(object):
	
	def get_frame(self):
		global cap1
		global df1
		cap1 = WebcamVideoStream(src=0).start()
		image = cap1.read()
		image=cv2.resize(image,(600,500))
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		face_rects=face_cascade.detectMultiScale(gray,1.3,5)
		# df1 = pd.read_csv(music_dist[show_text[0]])
		# df1 = df1[['Name','Album','Artist']]
		# df1 = df1.head(15)
		for (x,y,w,h) in face_rects:
			cv2.rectangle(image,(x,y-50),(x+w,y+h+10),(0,255,0),2)
			roi_gray_frame = gray[y:y + h, x:x + w]
			cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
			prediction = emotion_model.predict(cropped_img)
			maxindex = int(np.argmax(prediction))
			re.my_emotion = emotion_dict[maxindex]
			show_text[0] = maxindex 
			#print("===========================================",music_dist[show_text[0]],"===========================================")
			#print(df1)
			cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
			df1 = music_rec()
			print("predicted Emotion=",re.my_emotion)
			# if re.my_emotion != "neutral":
			# 	stop_camera()
			# 	break
			
			
		global last_frame1
		last_frame1 = image.copy()
		pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
		img = Image.fromarray(last_frame1)
		img = np.array(img)
		ret, jpeg = cv2.imencode('.jpg', img)

		return jpeg.tobytes(), df1
	
	def spotifyMusic(self):

		print("In spotify func")
		spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id='e6b7fe6fb4b74d13a55ace0c80af468e',client_secret='013532b07d1646ce845720be522acfc5'))

		if re.my_emotion == 'Happy':
			playlist = spotify.playlist_items('4nd7oGDNgfM0rv28CQw9WQ')
			
		elif re.my_emotion == "Sad":
			playlist = spotify.playlist_items('0z5GPu1ZL2ryEmPbTyH0tB')

		elif re.my_emotion == "Angry":
			playlist = spotify.playlist_items('0a4Hr64HWlxekayZ8wnWqx')
			
		else:
			playlist = spotify.playlist_items('3FvHenPTMZIv6Dw8elwKd7')


			# filter out records with no name
		tracks = [track for track in playlist['items'] if track['track'] and track['track']['name']]
			# show only the first 20 tracks
		tracks = tracks[:20]
		context = {'tracks': tracks}
		return context

def music_rec():
	print('---------------- Value ------------', music_dist[show_text[0]])
	# df = pd.read_csv(music_dist[show_text[0]])
	# df = df[['Name','Album','Artist']]
	# df = df.head(15)
	#return df

	

