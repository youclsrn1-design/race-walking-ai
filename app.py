import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Rule 54 Extension VAR", layout="wide")
st.title("🚨 Rule 54 공식 VAR (신전 구간 전용 판독)")
st.markdown("##### 💡 1. 굴곡된(접힌) 무릎은 무시하고, 다리가 '신전(뻗어진) 상태'일 때만 포착합니다.")
st.markdown("##### 💡 2. 신전된 다리 중 각도가 '170도 미만'인 경우만 파울로 캡처합니다.")
st.write("---")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    # 허리 중심(a) - 무릎 중심(b) - 종골(c) 사이의 각도 계산
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

st.error("⚠️ **10초 이내의 경보 영상**을 올려주세요. 뻗어진(신전된) 다리만 정밀 스캔합니다.")
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
    min_angle_in_step = 170.0 # 170도 밑으로 떨어질 때만 기록
    
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
            
        required_flight_frames = int(0.08 * fps)
        if required_flight_frames < 2: required_flight_frames = 2

        with st.spinner(f"🕵️‍♂️ 영상 분석 중... (굴곡된 다리 무시, '신전된 다리'만 필터링하여 감시 중)"):
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
                    # 🚨 [2단계] "신전된 다리(Extended Leg)" 필터링 및 170도 판독
                    # =========================================================
                    f_heel = l_heel if front_is_left else r_heel
                    f_knee = l_k if front_is_left else r_k
                    
                    is_in_support_phase = False
                    
                    if moving_right and f_heel[0] > waist_center[0]:
                        is_in_support_phase = True
                    elif not moving_right and f_heel[0] < waist_center[0]:
                        is_in_support_phase = True

                    if is_in_support_phase:
                        current_angle = calculate_angle(waist_center, f_knee, f_heel)
                        
                        # 💡 [핵심 알고리즘 수정] 신전 필터 (Extension Filter) 적용
                        # 다리가 150도 이상 펴진 '신전 상태'일 때만 분석! 
                        # 150도 미만으로 접힌 '굴곡 상태'는 스윙(Swing)이거나 착지 전이므로 철저히 무시
                        if current_angle >= 150.0: 
                            
                            # 신전되었지만 170도에 못 미칠 경우 (진짜 파울 구간)
                            if current_angle < 170.0:
                                # 그 중에서도 가장 심하게 굽혀진 순간 1장을 박제
                                if current_angle < min_angle_in_step:
                                    min_angle_in_step = current_angle
                                    
                                    line_thick = max(4, int(w / 180))
                                    cv2.line(annotated, tuple(waist_center), tuple(f_knee), (0, 0, 255), line_thick, cv2.LINE_AA)
                                    cv2.line(annotated, tuple(f_knee), tuple(f_heel), (0, 0, 255), line_thick, cv2.LINE_AA)
                                    
                                    cv2.circle(annotated, tuple(waist_center), 8, (255, 0, 255), -1) 
                                    cv2.circle(annotated, tuple(f_knee), 8, (0, 255, 255), -1)      
                                    cv2.circle(annotated, tuple(f_heel), 8, (0, 255, 0), -1)        
                                    
                                    cv2.putText(annotated, f"ANGLE: {current_angle:.1f}", (f_knee[0] + 20, f_knee[1]), 
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
        st.header("📸 Rule 54 신전 구간(Extension) VAR 리포트")
        
        st.subheader(f"🔴 Bent Knee (신전 170도 미만 포착): 총 {len(photo_finish_frames)}회 적발")
        if len(photo_finish_frames) > 0:
            st.error("⚠️ 다리가 '신전(뻗어짐)' 상태에 진입했으나, 170도를 채우지 못하고 굽혀진 채로 통과한 프레임들입니다.")
            for i in range(0, len(photo_finish_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(photo_finish_frames):
                        foul = photo_finish_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"신전 미달 파울 #{i+j+1} (각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 신전된 다리가 170도 이상을 훌륭하게 유지했습니다.")
            
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
st.info("💡 **신전 필터 가동 중:** 150도 미만으로 굽어있는(굴곡) 상태는 모두 무시하고, 다리가 150도 이상 펴졌음에도 170도를 넘지 못한 '불완전 신전' 파울만 캡처합니다.")
