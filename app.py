import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 페이지 설정 (전문가용 랩 느낌)
st.set_page_config(page_title="Elite Racewalking Analysis", page_icon="🏅")
st.title("🏅 Elite Racewalking Analysis Report")
st.caption("AI가 영상 내 가장 중요한 '착지 순간'을 자동으로 포착하여 분석합니다.")

# MediaPipe 초기화 (오류 방지용 설정 추가)
mp_pose = mp.solutions.pose
# model_complexity를 1로 낮추면 권한 오류 해결에 도움이 되고 속도가 빨라집니다.
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360-deg if deg > 180 else deg

# --- 영상 업로드 (유일한 입력 단계) ---
video_file = st.file_uploader("분석할 훈련 영상을 업로드하세요", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    key_frame_data = None
    max_knee_extension = 0

    with st.spinner("AI가 생체역학적 핵심 지표를 산출 중입니다..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img)
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # 관절 좌표 (왼쪽 기준)
                l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                l_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                l_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                l_foot = [lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]

                knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                ankle_angle = calculate_angle(l_knee, l_ankle, l_foot)

                # '무릎이 가장 펴진 순간'을 핵심 장면으로 자동 포착
                if knee_angle > max_knee_extension:
                    max_knee_extension = knee_angle
                    annotated_img = img.copy()
                    mp_drawing.draw_landmarks(annotated_img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    key_frame_data = {"image": annotated_img, "knee": knee_angle, "ankle": ankle_angle}

    cap.release()
    os.unlink(tfile.name)

    # --- 핵심 결과 리포트 ---
    if key_frame_data:
        st.subheader("📍 핵심 생체역학 분석 결과")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(key_frame_data["image"], caption="포착된 핵심 착지 모멘트", use_container_width=True)
        
        with col2:
            # 고객에게 필요한 핵심 수치만 노출
            st.metric("무릎 신전도 (Knee)", f"{key_frame_data['knee']:.1f}°")
            st.metric("발목 착지각 (Ankle)", f"{key_frame_data['ankle']:.1f}°")
            
            if key_frame_data["knee"] >= 175:
                st.success("✅ Rule 54 준수")
            else:
                st.error("⚠️ 무릎 신전 경고")

        st.markdown("---")
        st.subheader("📋 전문 생체역학 처방")
        a_angle = key_frame_data["ankle"]
        if a_angle < 23:
            st.warning("**[처방]** 발목 각도가 낮아 제동력이 발생합니다. 발가락을 더 당겨서 착지하세요.")
        elif 23 <= a_angle <= 27:
            st.info("**[처방]** 이상적인 발목 각도입니다. 보폭 확장에 집중하세요.")
        else:
            st.warning("**[처방]** 발목이 너무 높습니다. 착지 시 불안정할 수 있습니다.")
