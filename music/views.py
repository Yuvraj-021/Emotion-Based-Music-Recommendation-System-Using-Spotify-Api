import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from django.http import HttpResponse
from django.shortcuts import render

import pandas as pd
import csv

from recognition.response import StreamingHttpResponse
from recognition.test import VideoCamera
from recognition.test import WebcamVideoStream
import threading
import cv2
import recognition.Emotion as re
from django.shortcuts import render, redirect
from django.http import JsonResponse


def home(request):
    return render(request, "index.html")


def blog(request):
    return render(request, "blog.html")


def about(request):
    return render(request, "about.html")


def profile(request):
    return render(request, "profile.html")


streaming = True
# def facecam_feed(request):
# try:
# cam = VideoCamera()
# return StreamingHttpResponse(gen_frames(), content_type="multipart/x-mixed-replace;boundary=frame")
# except:
#     pass
# return render(request, "index.html")


def gen(camera):
    # i=1
    while streaming:
        global df1
        frame, df1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def facecam_feed(self):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type="multipart/x-mixed-replace;boundary=frame")


def stop_emotion_detection(request):
    my_instance = WebcamVideoStream()
    my_instance.stop()
    # return HttpResponse('Emotion detection stopped')
    print("Emotion detection stopped")
    return render(request, 'index.html')


def stop_streaming(request):
    global streaming
    streaming = False
    return JsonResponse({'status': 'ok'})


def spotifyApi(request):
    birdy_uri = 'spotify:playlist:0z5GPu1ZL2ryEmPbTyH0tB'
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id='e6b7fe6fb4b74d13a55ace0c80af468e', client_secret='013532b07d1646ce845720be522acfc5'))

    if re.my_emotion == 'Happy':
        playlist = spotify.playlist_items('4nd7oGDNgfM0rv28CQw9WQ')

    elif re.my_emotion == "Sad":
        playlist = spotify.playlist_items('0z5GPu1ZL2ryEmPbTyH0tB')

    elif re.my_emotion == "Angry":
        playlist = spotify.playlist_items('0a4Hr64HWlxekayZ8wnWqx')

    else:
        playlist = spotify.playlist_items('3FvHenPTMZIv6Dw8elwKd7')

    # filter out records with no name
    tracks = [track for track in playlist['items']
              if track['track'] and track['track']['name']]
    # show only the first 20 tracks
    tracks = tracks[:]
    context = {'tracks': tracks}

    # context=VideoCamera.spotifyMusic()
    return render(request, 'spotify.html', context)


################################################################################
    # # Get the tracks in the playlist from the Spotify API
    # tracks = spotify.playlist_tracks('0z5GPu1ZL2ryEmPbTyH0tB')['items']

    # # Render the template with the track data
    # return render(request, 'spotify.html', {'tracks': tracks})
#####################################################################################
    # Commented now
    # results = spotify.playlist("0z5GPu1ZL2ryEmPbTyH0tB")
    # for track in results['tracks']['items']:
    #     print(track['track']['name'])

    # return render(request,"spotify.html",{"album":results})


# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         (self.grabbed, self.frame) = self.video.read()
#         threading.Thread(target=self.update, args=()).start()

#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         image = self.frame
#         _, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()

#     def update(self):
#         while True:
#             (self.grabbed, self.frame) = self.video.read()
