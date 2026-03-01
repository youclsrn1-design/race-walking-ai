import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 페이지 설정 (전문가용 랩 느낌)
st.set_page_config(page_title="Elite Racewalking Analysis", page_icon="🏅")
st.title("🏅 Elite Racewalking Analysis Report")
st.markdown("---")

# MediaPipe 내부 설정 (고객에게 노출 안 함)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360-deg if deg > 180 else deg

# 1. 영상 업로드 (유일한 입력 단계)
video_file = st.file_uploader("분석할 훈련 영상을 업로드하세요", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # 핵심 프레임 추출을 위한 변수
    key_frame_data = None
    max_knee_extension = 0

    with st.spinner("AI가 생체역학적 핵심 지표를 산출하고 있습니다..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img)
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                
                # 필요 관절 좌표 추출
                l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                l_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                l_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                l_foot = [lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]

                # 핵심 각도 계산
                knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                ankle_angle = calculate_angle(l_knee, l_ankle, l_foot)

                # '무릎이 가장 펴진 순간(착지 순간)'을 핵심 프레임으로 포착
                if knee_angle > max_knee_extension:
                    max_knee_extension = knee_angle
                    # 분석용 시각화 처리
                    annotated_img = img.copy()
                    mp_drawing.draw_landmarks(annotated_img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    key_frame_data = {
                        "image": annotated_img,
                        "knee": knee_angle,
                        "ankle": ankle_angle
                    }

    cap.release()
    os.unlink(tfile.name)

    # 2. 결과 리포트 (핵심 정보만 노출)
    if key_frame_data:
        st.subheader("📍 핵심 생체역학 분석 결과")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(key_frame_data["image"], caption="포착된 핵심 착지 모멘트", use_container_width=True)
        
        with col2:
            # 주요 지표를 직관적으로 표시
            st.metric("무릎 신전도 (Knee Extension)", f"{key_frame_data['knee']:.1f}°")
            st.metric("발목 착지각 (Ankle Angle)", f"{key_frame_data['ankle']:.1f}°")
            
            # 규칙 준수 여부 판정
            if key_frame_data["knee"] >= 175:
                st.success("✅ Rule 54 준수 (무릎 완전 신전)")
            else:
                st.error("⚠️ 무릎 신전 부족 (경고 위험)")

        # 3. 전문가 처방 (Prescription)
        st.markdown("---")
        st.subheader("📋 전문 생체역학 처방")
        
        # 발목 각도에 따른 맞춤형 처방 로직
        a_angle = key_frame_data["ankle"]
        if a_angle < 23:
            st.warning("**[처방]** 발목 각도가 너무 낮습니다. 지면 제동력이 발생하여 전진 속도가 감소합니다. 발가락을 조금 더 정강이 쪽으로 당겨(Dorsiflexion) 착지하세요.")
        elif 23 <= a_angle <= 27:
            st.info("**[처방]** 이상적인 발목 각도입니다. 현재의 착지 메커니즘을 유지하면서 보폭(Stride length)을 확장하는 데 집중하세요.")
        else:
            st.warning("**[처방]** 발목 각도가 너무 높습니다. 족저굴곡이 과도하여 착지 시 불안정성이 생길 수 있습니다.")
