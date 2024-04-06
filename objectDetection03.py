from IPython.utils.sysinfo import num_cpus
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO


# アップロードした画像から人数を計測する

# read model
model = YOLO("yolov8n.pt")

# input from file uploader
uploaded_img = st.file_uploader("Chose a file", type=["png", "jpg"])

# process
if uploaded_img is not None:
    # To read image file buffer with OpenCV:
    bytes_data = uploaded_img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # モデルによる物体検知の予測を人の検知に絞る
    # confは検知確率の閾値を設定
    results = model(cv2_img, conf=0.5, classes=[0])
    # 予測結果のプロット
    output_img = results[0].plot(labels=True, conf=True)
    # OpencvではカラーがBGR表記になっているので、画像を出力するためにRGBに変更
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # 人数を計測
    cls = results[0].boxes.cls
    num_people = len(cls)

    # output
    st.image(output_img, caption="画像出力")
    st.text(f"{num_people}人")