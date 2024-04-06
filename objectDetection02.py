import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

#PC内蔵カメラによるインプットから物体検知

# read model
model = YOLO("yolov8n.pt")

# input
img_file_buffer = st.camera_input("Take a picture")

# process
if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # モデルによる物体検知の予測
    # confは検知確率の閾値を設定
    results = model(cv2_img, conf=0.5)
    # 予測結果のプロット
    output_img = results[0].plot(labels=True, conf=True)
    # OpencvではカラーがBGR表記になっているので、画像を出力するためにRGBに変更
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # output
    st.image(output_img, caption="画像出力")