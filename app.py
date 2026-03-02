import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Rule 54 Heel Strike VAR", layout="wide")
st.title("🚨 Rule 54 공식 VAR (종골 착지점 기준 판독)")
st.markdown("##### 💡 1. 허공에 뜬 다리는 무시하고, 종골(뒤꿈치)이 '지면에 닿는 순간'부터 감시를 시작합니다.")
st.markdown("##### 💡 2. 지면에 닿은 앞발이 170도 미만으로 굽혀지면 즉각 파울로 캡처합니다.")
st.write("---")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

st.error("⚠️ **10초 이내의 경보 영상**을 올려주세요. 종골이 땅에 닿는 '착지점'부터 정밀 스캔합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    photo_finish_frames = []   
    flight_foul_frames = []
    
    global_ground_y = 0.0
    flight_frames_count = 0 
    
    prev_front_leg = None
    worst_foul_frame = None
    min_angle_in_step = 170.0 
    
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
            
        required_flight_frames = int(0.08 * fps)
        if required_flight_frames < 2: required_flight_frames = 2

        with st.spinner(f"🕵️‍♂️ 영상 분석 중... (허공 스윙 구간 무시, '종골 착지점'부터 추적 중)"):
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
                    l_heel = get_pt(lm[mp_pose.PoseLandmark.LEFT_HEEL])
                    r_heel = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HEEL])
                    nose_x = lm[mp_pose.PoseLandmark.NOSE].x * w
                    
                    waist_center = [int((l_h[0] + r_h[0]) / 2), int((l_h[1] + r_h[1]) / 2)]
                    moving_right = nose_x > waist_center[0]
                    
                    if moving_right:
                        front_is_left = l_heel[0] > r_heel[0]
                    else:
                        front_is_left = l_heel[0] < r_heel[0]
                        
                    current_front_leg = 'L' if front_is_left else 'R'

                    # 실시간 최저 지면(Ground) 업데이트
                    current_lowest_y = max(l_heel[1], r_heel[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y
                        
                    annotated = img.copy()

                    # =========================================================
                    # 🚨 [1단계] 걸음(Step) 전환 감지 및 초기화
                    # =========================================================
                    if current_front_leg != prev_front_leg:
                        if worst_foul_frame is not None:
                            photo_finish_frames.append(worst_foul_frame)
                        worst_foul_frame = None
                        min_angle_in_step = 170.0
                        prev_front_leg = current_front_leg

                    # =========================================================
                    # 🚨 [2단계] "착지한 앞발"만 타겟팅 (Heel Strike Filter)
                    # =========================================================
                    f_heel = l_heel if front_is_left else r_heel
                    f_knee = l_k if front_is_left else r_k
                    
                    # 1. 앞발이 허리보다 앞에 있는가?
                    is_in_front = (f_heel[0] > waist_center[0]) if moving_right else (f_heel[0] < waist_center[0])
                    
                    # 💡 2. 앞발 종골이 땅에 닿았는가? (오차범위: 화면 높이의 4% 이내로 근접)
                    is_grounded = abs(global_ground_y - f_heel[1]) < (h * 0.04)

                    # 앞발이 앞에 있고 + 종골이 땅에 닿은 '지지 구간(Support Phase)'일 때만 판독 시작!
                    if is_in_front and is_grounded:
                        current_angle = calculate_angle(waist_center, f_knee, f_heel)
                        
                        # 지면에 닿았는데 170도를 채우지 못했다면 파울 기록 갱신
                        if current_angle < 170.0:
                            if current_angle < min_angle_in_step:
                                min_angle_in_step = current_angle
                                
                                line_thick = max(4, int(w / 180))
                                cv2.line(annotated, tuple(waist_center), tuple(f_knee), (0, 0, 255), line_thick, cv2.LINE_AA)
                                cv2.line(annotated, tuple(f_knee), tuple(f_heel), (0, 0, 255), line_thick, cv2.LINE_AA)
                                
                                cv2.circle(annotated, tuple(waist_center), 8, (255, 0, 255), -1) 
                                cv2.circle(annotated, tuple(f_knee), 8, (0, 255, 255), -1)      
                                cv2.circle(annotated, tuple(f_heel), 8, (0, 255, 0), -1)        
                                
                                cv2.putText(annotated, f"GROUNDED ANGLE: {current_angle:.1f}", (f_knee[0] + 20, f_knee[1]), 
                                            cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w/900), (0, 255, 255), 3)
                                cv2.putText(annotated, "BENT KNEE (FAILED EXTENSION)", (30, 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                                            
                                worst_foul_frame = (current_angle, annotated.copy())

                    # =========================================================
                    # 🚨 [룰 3] 체공 시간 측정: 0.08초 초과
                    # =========================================================
                    flight_gap = global_ground_y - current_lowest_y
                    if flight_gap > (h * 0.02): 
                        flight_frames_count += 1
                    else:
                        if flight_frames_count >= required_flight_frames:
                            flight_time_sec = flight_frames_count / fps
                            cv2.line(annotated, (0, int(global_ground_y)), (w, int(global_ground_y)), (0, 255, 0), 2)
                            cv2.putText(annotated, f"FLIGHT FOUL: {flight_time_sec:.3f}s (>0.08s)", (20, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                            flight_foul_frames.append(annotated.copy())
                        flight_frames_count = 0

            if worst_foul_frame is not None:
                photo_finish_frames.append(worst_foul_frame)

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 최종 리포트
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("📸 Rule 54 종골 착지 지점 VAR 리포트")
        
        st.subheader(f"🔴 Bent Knee (지면 착지 시 170도 미만): 총 {len(photo_finish_frames)}회 적발")
        if len(photo_finish_frames) > 0:
            st.error("⚠️ 종골(뒤꿈치)이 지면에 닿은 순간부터 수직이 될 때까지의 구간에서, 다리가 170도 미만으로 굽혀진 프레임들입니다.")
            for i in range(0, len(photo_finish_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(photo_finish_frames):
                        foul = photo_finish_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"착지 파울 #{i+j+1} (각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 종골이 땅에 닿는 순간부터 수직이 될 때까지 170도 이상 완벽하게 신전되었습니다.")
            
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
st.info("💡 **지면 착지 감지 가동 중:** 허공에서 다리를 뻗거나 스윙하는 동작은 철저히 무시하며, 오직 **앞발 종골(뒤꿈치)이 땅에 닿아있는 상태**에서만 170도 미만의 굽힘을 적발합니다.")
