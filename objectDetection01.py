import streamlit as st
import cv2
import numpy as np

# PC内蔵カメラによるインプット

# input
img_file_buffer = st.camera_input("Take a picture")

# process
if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # OpencvではカラーがBGR表記になっているので、画像を出力するためにRGBに変更
    output_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # output
    st.image(output_img, caption="画像出力")