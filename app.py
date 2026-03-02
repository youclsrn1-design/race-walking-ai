import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Foul Catcher", layout="wide")
st.title("🚨 국제 심판 전용: Rule 54 파울 적발 AI")
st.markdown("##### 💡 오직 'Bent Knee(무릎 굽힘)'와 'Loss of Contact(양발 체공)' 파울만 추적하여 증거 프레임을 화면에 박제합니다.")
st.write("---")

# 2. AI 분석 엔진
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

# 3. 영상 업로드
st.error("⚠️ **10초 이내의 영상**을 올려주세요.")
video_file = st.file_uploader("경보 역학 분석 영상 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    foul_bent_knee_frames = []
    foul_contact_frames = []
    
    prev_stride_dist = 0
    prev_trend = 0
    ground_y = 0.0
    
    cd_bent_knee = 0
    cd_contact = 0
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("🕵️‍♂️ AI 국제 심판이 영상을 분석하여 파울 프레임을 캡처 중입니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                h, w = frame.shape[:2]
                if w > 640: frame = cv2.resize(frame, (640, int(h * 640 / w)))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                if cd_bent_knee > 0: cd_bent_knee -= 1
                if cd_contact > 0: cd_contact -= 1
                
                if res.pose_landmarks:
                    person_detected = True
                    lm = res.pose_landmarks.landmark
                    
                    l_h = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                    r_h = [lm[mp_
