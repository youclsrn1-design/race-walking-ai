import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
from collections import Counter
from PIL import Image

# 1. 전문가용 인터페이스 설정
st.set_page_config(page_title="World Athletics AI Judge", layout="wide")
st.title("🔬 Pro Racewalking AI Judge (스마트 필터링 엔진)")
st.markdown("##### AI가 영상의 시점(측면/정면)을 자동 파악하여, 쓸데없는 데이터(오류)를 버리고 진짜 '착지 구간'만 정밀 판독합니다.")
st.warning("🔒 업로드된 영상은 분석 직후 즉시 영구 삭제되어 서버에 남지 않습니다.")
st.write("---")

# 2. AI 분석 엔진 초기화
mp_pose = mp.solutions.pose
# 3D 뎁스(Z축)까지 정밀하게 읽어오기 위해 세팅
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360-deg if deg > 180 else deg

def get_tilt_angle(p1, p2):
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    return np.abs(np.degrees(np.arctan2(dy, dx)))

# 3. 통합 영상 업로드 및 분석
st.subheader("영상 업로드")
st.error("⚠️ **10초 이내의 영상**만 올려주세요! AI가 구도(측면/정면)를 스스로 판단합니다.")
video_file = st.file_uploader("경보 훈련 영상 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    view_types = [] # 영상의 시점을 담을 리스트
    knee_stats = [] # 무릎 각도 모음 (측면용)
    hip_tilt_stats = [] # 골반 기울기 모음 (정면용)
    
    key_frame_side = None
    key_frame_front = None
    max_knee = 0
    max_hip_tilt = 0
    frame_count = 0
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("AI가 영상의 시점을 파악하고, 허수 데이터를 걸러내는 중입니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                if frame_count % 3 != 0: continue 
                
                h, w = frame.shape[:2]
                if w > 640: frame = cv2.resize(frame, (640, int(h * 640 / w)))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                if res.pose_landmarks:
                    person_detected = True
                    lm = res.pose_landmarks.landmark
                    
                    # 좌우 어깨, 골반 좌표 추출
                    l_s = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    r_s = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                    l_h = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                    r_h = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                    
                    # 💡 [핵심 기술 1] 시점 자동 인식 로직
                    # 어깨 너비와 몸통 길이를 비교하여 측면인지 정면인지 AI가 판단
                    shoulder_width = abs(l_s[0] - r_s[0])
                    torso_height = abs((l_s[1] + r_s[1])/2 - (l_h[1] + r_h[1])/2)
                    ratio = shoulder_width / (torso_height + 0.0001)
                    
                    # 비율이 0.35보다 크면 정면(어깨가 넓게 보임), 작으면 측면(어깨가 겹쳐 보임)
                    current_view = "front" if ratio > 0.35 else "side"
                    view_types.append(current_view)

                    # 카메라에 더 가까운 다리(Z축 활용)를 자동으로 타겟팅
                    l_a_z = lm[mp_pose.PoseLandmark.LEFT_ANKLE].z
                    r_a_z = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].z
                    
                    if l_a_z < r_a_z: # 왼쪽이 더 가까움
                        hip = l_h
                        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                        foot = [lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
                    else:             # 오른쪽이 더 가까움
                        hip = r_h
                        knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                        ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
                        foot = [lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]

                    annotated = img.copy()
                    mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    safe_img = annotated.copy()

                    # 측면 역학 계산
                    k_ang = calculate_angle(hip, knee, ankle)
                    a_ang = calculate_angle(knee, ankle, foot)
                    knee_stats.append(k_ang)

                    if k_ang > max_knee:
                        max_knee = k_ang
                        key_frame_side = {"img": safe_img, "k": k_ang, "a": a_ang}

                    # 정면 역학 계산
                    tilt = get_tilt_angle(l_h, r_h)
                    hip_tilt_stats.append(tilt)

                    if tilt > max_hip_tilt:
                        max_hip_tilt = tilt
                        key_frame_front = {"img": safe_img, "tilt": tilt}

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    if not person_detected:
        st.error("❌ 영상에서 선수를 인식하지 못했습니다.")
    else:
        # 영상의 최종 시점 확정 (가장 많이 감지된 구도)
        dominant_view = Counter(view_types).most_common(1)[0][0] if view_types else "side"

        st.divider()
        st.subheader("🎯 AI 시점 자동 인식 결과")
        
        if dominant_view == "side":
            st.success("🎥 **[측면 영상 감지]** 오류 방지를 위해 정면 골반 밸런스(Hip Drop) 분석을 차단하고, Rule 54(무릎 신전) 정밀 판독 모드를 가동합니다.")
            
            # 💡 [핵심 기술 2] 스마트 평균 계산 (무릎이 구부러진 스윙 구간은 평균에서 버림)
            contact_knees = [k for k in knee_stats if k >= 160] # 160도 이상 펴진 프레임만 착지로 간주
            avg_knee = np.mean(contact_knees) if contact_knees else max_knee

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(key_frame_side["img"], channels="RGB", use_column_width=True)
            with col2:
                st.markdown("#### 🏃‍♂️ Rule 54 착지 역학")
                st.metric("최대 무릎 신전도", f"{key_frame_side['k']:.1f}°")
                st.metric("착지 순간 발목 각도", f"{key_frame_side['a']:.1f}°")
                st.markdown(f"**실제 착지 구간 평균 무릎 각도: {avg_knee:.1f}°**")
                st.caption("(무릎이 접히는 스윙 구간의 허수 데이터는 제외되었습니다.)")

                if key_frame_side["k"] >= 178:
                    st.success("🔥 **[통과] 완벽한 무릎 강체 (Rigid lever)를 유지 중입니다.**")
                else:
                    st.error("⚠️ **[경고] 벤트 니(Bent Knee) 위반 위험이 있습니다.**")

        else: # dominant_view == "front"
            st.success("🎥 **[정면 영상 감지]** 측면 무릎 파울 판독을 차단하고, 골반 좌우 밸런스(Hip Drop) 및 코어 안정성 판독 모드를 가동합니다.")
            
            avg_tilt = np.mean(hip_tilt_stats) if hip_tilt_stats else 0
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(key_frame_front["img"], channels="RGB", use_column_width=True)
            with col2:
                st.markdown("#### ⚖️ 코어 밸런스 역학")
                st.metric("최대 골반 기울기 (Hip Drop)", f"{key_frame_front['tilt']:.1f}°")
                st.markdown(f"**평균 골반 기울기: {avg_tilt:.1f}°**")
                
                tilt_val = key_frame_front["tilt"]
                if tilt_val < 5:
                    st.error("⚠️ **[리듬 결여]** 골반이 통나무처럼 굳어있어 충격이 어깨로 올라옵니다.")
                elif 5 <= tilt_val <= 12:
                    st.success("🔥 **[리듬 최적]** 물 흐르듯 완벽한 시계추 리듬(Pendulum rhythm)이 형성 중입니다.")
                else:
                    st.warning("⚠️ **[에너지 누수]** 골반이 12도 이상 무너집니다. 중둔근 코어 강화가 필요합니다.")

st.write("---")
st.info("💡 **시스템 개선을 위한 소중한 의견을 들려주세요!**\n\n버그 신고, 추가 판독 지표 제안 등 어떠한 피드백이든 적극 수용합니다.\n\n📧 **문의처:** [youclsrn1@gmail.com](mailto:youclsrn1@gmail.com)")
