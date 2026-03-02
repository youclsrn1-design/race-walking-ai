import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Official VAR", layout="wide")
st.title("🚨 Rule 54 공식 VAR (무릎 중심축 판독 엔진)")
st.markdown("##### 💡 1. 무릎 앞면과 뒷면의 '정중앙 관절축'을 관통하는 빨간선으로 170도 이하 신전 실패(Bent Knee)를 잡아냅니다.")
st.markdown("##### 💡 2. 두 발이 지면에서 '0.08초' 이상 떨어지면 즉각 실격 (Loss of Contact)")
st.write("---")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    # 골반(a) - 무릎 중심축(b) - 발목(c) 사이의 내각 계산
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

st.error("⚠️ **10초 이내의 훈련 영상**을 올려주세요. 무릎 관절 중심축을 스캔합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    photo_finish_frames = []   
    flight_foul_frames = []
    
    prev_stride_dist = 0
    prev_trend = 0
    prev_hip_x = 0
    
    global_ground_y = 0.0
    flight_frames_count = 0 
    
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
            
        required_flight_frames = int(0.08 * fps)
        if required_flight_frames < 2: required_flight_frames = 2

        with st.spinner(f"🕵️‍♂️ 영상 분석 중... 무릎 앞면과 뒷면의 '중심축'을 스캔하고 있습니다."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                h, w = frame.shape[:2]
                if w > 800: frame = cv2.resize(frame, (800, int(h * 800 / w)))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                if res.pose_landmarks:
                    person_detected = True
                    lm = res.pose_landmarks.landmark
                    
                    def get_pt(landmark):
                        return [int(landmark.x * w), int(landmark.y * h)]
                    
                    l_h = get_pt(lm[mp_pose.PoseLandmark.LEFT_HIP])
                    r_h = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                    l_k = get_pt(lm[mp_pose.PoseLandmark.LEFT_KNEE])
                    r_k = get_pt(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                    l_a = get_pt(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
                    r_a = get_pt(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                    nose_x = lm[mp_pose.PoseLandmark.NOSE].x * w
                    
                    hip_center_x = (l_h[0] + r_h[0]) / 2
                    if prev_hip_x == 0: prev_hip_x = hip_center_x
                    moving_right = hip_center_x > prev_hip_x
                    
                    current_lowest_y = max(l_a[1], r_a[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y
                        
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    annotated = img.copy()

                    # =========================================================
                    # 🚨 [룰 1] 착지 순간: 골반 - '무릎 중심축' - 발목 신전 각도
                    # =========================================================
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1):
                        leading_is_left = (l_a[0] > r_a[0]) if moving_right else (l_a[0] < r_a[0])

                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        
                        front_angle = calculate_angle(front_hip, front_knee, front_ankle)
                        
                        if front_angle <= 170.0:
                            line_thick = max(4, int(w / 180))
                            
                            # 1. 골반 -> 무릎 중심축 빨간선
                            cv2.line(annotated, tuple(front_hip), tuple(front_knee), (0, 0, 255), line_thick, cv2.LINE_AA)
                            # 2. 무릎 중심축 -> 발목 빨간선
                            cv2.line(annotated, tuple(front_knee), tuple(front_ankle), (0, 0, 255), line_thick, cv2.LINE_AA)
                            
                            # 💡 3. 무릎 중심축 타겟 시각화 (앞면과 뒷면의 중앙임을 보여주는 타겟 링)
                            center_radius = max(5, int(w/180))
                            outer_radius = max(12, int(w/80))
                            # 내부 정중앙 점 (청록색)
                            cv2.circle(annotated, tuple(front_knee), center_radius, (255, 255, 0), -1, cv2.LINE_AA)
                            # 무릎 두께를 의미하는 외부 링 (청록색 테두리)
                            cv2.circle(annotated, tuple(front_knee), outer_radius, (255, 255, 0), 2, cv2.LINE_AA)
                            
                            cv2.putText(annotated, f"CENTRAL AXIS: {front_angle:.1f} deg", (front_knee[0] + 20, front_knee[1]), 
                                        cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w/900), (0, 255, 255), 3)
                            
                            photo_finish_frames.append((front_angle, annotated.copy()))

                    # =========================================================
                    # 🚨 [룰 2] 체공 시간 측정: 0.08초 초과
                    # =========================================================
                    flight_gap = global_ground_y - current_lowest_y
                    
                    if flight_gap > (h * 0.02): 
                        flight_frames_count += 1
                    else:
                        if flight_frames_count >= required_flight_frames:
                            flight_time_sec = flight_frames_count / fps
                            
                            cv2.line(annotated, (0, int(global_ground_y)), (w, int(global_ground_y)), (0, 255, 0), 2)
                            cv2.putText(annotated, f"FLIGHT FOUL: {flight_time_sec:.3f} sec (>0.08s)", (20, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                            
                            flight_foul_frames.append(annotated.copy())
                            
                        flight_frames_count = 0

                    prev_stride_dist = stride_dist
                    prev_trend = trend
                    prev_hip_x = hip_center_x

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 최종 리포트
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("📸 Rule 54 공식 VAR 판독 리포트")
        
        st.subheader(f"🔴 Bent Knee (무릎 중심축 170도 이하): 총 {len(photo_finish_frames)}회 적발")
        if len(photo_finish_frames) > 0:
            st.error("⚠️ 착지 순간 무릎 관절의 '정중앙 축(타겟 링)'을 관통하는 선이 170도 이하로 굽혀졌습니다.")
            for i in range(0, len(photo_finish_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(photo_finish_frames):
                        foul = photo_finish_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 #{i+j+1} (중심축 각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 전방 다리가 수직선을 통과할 때까지 중심축이 완벽하게 펴져 있습니다.")
            
        st.write("---")
        
        st.subheader(f"🟡 Loss of Contact (0.08초 초과 체공): 총 {len(flight_foul_frames)}회 적발")
        if len(flight_foul_frames) > 0:
            st.warning("⚠️ 두 발이 허공에 0.08초를 초과하여 머물렀습니다. (즉각 실격)")
            for i in range(0, len(flight_foul_frames), 3):
                cols2 = st.columns(3)
                for j in range(3):
                    if i + j < len(flight_foul_frames):
                        img = flight_foul_frames[i + j]
                        with cols2[j]:
                            st.image(img, channels="RGB", caption=f"0.08초 초과 비행 적발 #{i+j+1}")
        else:
            st.success("✅ Loss of Contact 통과: 체공 시간이 0.08초 이하로 규칙을 준수했습니다.")

st.write("---")
st.info("💡 **판독 기준:** 무릎 앞면과 뒷면의 '중앙 관절축'을 기준으로 빨간선을 그어 170도 신전 여부를 엄격하게 판독합니다.")
