import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd

# read model
model = YOLO("yolov8n.pt")

# input from file uploader
uploaded_vid = st.file_uploader("Chose a video", type="mp4")

# process
if uploaded_vid is not None:
  temp_file = tempfile.NamedTemporaryFile(delete=False)
  temp_file.write(uploaded_vid.read())

  # 動画ファイルの読み込み
  cap = cv2.VideoCapture(temp_file.name)
  # 動画の情報を抽出
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  fps = cap.get(cv2.CAP_PROP_FPS)

  writer = cv2.VideoWriter("./object_detection_04_results.mp4", 
  cv2.VideoWriter_fourcc(*"MP4V"),fps,frameSize=(int(width),int(height)))

  num = 0
  nums = []
  person = []
  while cap.isOpened(): #正常に取り込めているかのチェック
    if num > count :break
    # ret: 画像の取得が成功したかどうかの結果(True/Fales)
    # img: 画像データ
    ret, img = cap.read()

    if ret:
      results = model(img, conf=0.5, classes=[0])
      output_img = results[0].plot(labels=False, conf=True)
      cls = results[0].boxes.cls
      num_people = len(cls)
      writer.write(output_img)
    nums.append(num)
    person.append(num_people)
    num += 1
  cap.release()
  writer.release()

  person_data = pd.DataFrame({"frame": nums, "count": person})
  person_data["sec"] = person_data["frame"]/fps
  person_data = person_data[["sec","count"]]

# output
  st.line_chart(person_data, x="sec", y="count")
  st.dataframe(person_data)