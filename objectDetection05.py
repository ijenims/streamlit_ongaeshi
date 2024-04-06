import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from ultralytics import YOLO
import av

model = YOLO("yolov8n.pt")

#　エッジ抽出
# def callback(frame):
#     img = frame.to_ndarray(format="bgr24")

#     img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

#     return av.VideoFrame.from_ndarray(img, format="bgr24")

def callback(frame):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        
        img = frame.to_ndarray(format="bgr24")
    
        results = model(img, conf=0.5, classes=[0])
        img = results[0].plot(labels=True,conf=True)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="sample", video_frame_callback=callback)