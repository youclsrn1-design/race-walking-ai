import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Rule 54 Support Phase VAR", layout="wide")
st.title("🚨 Rule 54 공식 VAR (지지 구간 연속 추적 엔진)")
st.markdown("##### 💡 1. [종골 착지점] ~ [온몸 수직 상태]까지의 전 구간을 추적하여 무릎 붕괴를 잡아냅니다.")
st.markdown("##### 💡 2. 측정 축: '허리 중심축 - 무릎 중심 - 종골(뒤꿈치)'")
st.write("---")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    # 허리 중심(a) - 무릎 중심(b) - 종골(c) 사이의 각도 계산
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

st.error("⚠️ **10초 이내의 영상**을 올려주세요. 착지부터 수직 통과까지 다리 신전을 끝까지 추적합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    photo_finish_frames = []   
    flight_foul_frames = []
    
    prev_stride_dist = 0
    prev_trend = 0
    prev_waist_x = 0
    
    global_ground_y = 0.0
    flight_frames_count = 0 
    
    # 💡 [핵심] 지지 구간(Support Phase) 연속 추적을 위한 상태 변수
    tracking_phase = False
    current_front_leg = None # 'L' or 'R'
    min_angle_in_phase = 360.0
    worst_foul_frame = None
    
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
            
        required_flight_frames = int(0.08 * fps)
        if required_flight_frames < 2: required_flight_frames = 2

        with st.spinner(f"🕵️‍♂️ 영상 분석 중... (착지 후 온몸이 수직이 될 때까지 각도를 연속 추적합니다)"):
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
                    
                    # 관절 및 💡 종골(HEEL) 좌표 가져오기
                    l_h = get_pt(lm[mp_pose.PoseLandmark.LEFT_HIP])
                    r_h = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                    l_k = get_pt(lm[mp_pose.PoseLandmark.LEFT_KNEE])
                    r_k = get_pt(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                    l_heel = get_pt(lm[mp_pose.PoseLandmark.LEFT_HEEL])
                    r_heel = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HEEL])
                    
                    # 허리 중심축 (골반 사이 정중앙)
                    waist_center = [int((l_h[0] + r_h[0]) / 2), int((l_h[1] + r_h[1]) / 2)]
                    
                    if prev_waist_x == 0: prev_waist_x = waist_center[0]
                    moving_right = waist_center[0] > prev_waist_x
                    
                    # 종골(뒤꿈치) 기준 지면 높이 및 보폭
                    current_lowest_y = max(l_heel[1], r_heel[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y
                        
                    stride_dist = abs(l_heel[0] - r_heel[0])
                    trend = stride_dist - prev_stride_dist
                    annotated = img.copy()

                    # =========================================================
                    # 🚨 [1단계] 배측굴곡 상태의 앞다리 착지 순간 포착 (추적 시작)
                    # =========================================================
                    if not tracking_phase and trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1):
                        tracking_phase = True # 💡 추적 레이더 ON
                        
                        if moving_right:
                            current_front_leg = 'L' if l_heel[0] > r_heel[0] else 'R'
                        else:
                            current_front_leg = 'L' if l_heel[0] < r_heel[0] else 'R'
                            
                        min_angle_in_phase = 360.0
                        worst_foul_frame = None

                    # =========================================================
                    # 🚨 [2단계] 착지점 ~ 온몸이 수직이 될 때까지 연속 각도 측정
                    # =========================================================
                    if tracking_phase:
                        f_heel = l_heel if current_front_leg == 'L' else r_heel
                        f_knee = l_k if current_front_leg == 'L' else r_k
                        
                        # 각도 계산 (허리중심 - 무릎중심 - 종골)
                        current_angle = calculate_angle(waist_center, f_knee, f_heel)
                        
                        line_thick = max(4, int(w / 180))
                        # 허리중심 -> 무릎중심 선
                        cv2.line(annotated, tuple(waist_center), tuple(f_knee), (0, 0, 255), line_thick, cv2.LINE_AA)
                        # 무릎중심 -> 종골(디딤점) 선
                        cv2.line(annotated, tuple(f_knee), tuple(f_heel), (0, 0, 255), line_thick, cv2.LINE_AA)
                        
                        # 축 시각화
                        cv2.circle(annotated, tuple(waist_center), 8, (255, 0, 255), -1) # 허리 축 (보라)
                        cv2.circle(annotated, tuple(f_knee), 8, (0, 255, 255), -1)      # 무릎 축 (노랑)
                        cv2.circle(annotated, tuple(f_heel), 8, (0, 255, 0), -1)        # 종골 디딤점 (초록)
                        
                        cv2.putText(annotated, f"ANGLE: {current_angle:.1f}", (f_knee[0] + 20, f_knee[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w/900), (0, 255, 255), 3)

                        # 가장 굽혀진 순간(최저 각도) 캡처 갱신
                        if current_angle < min_angle_in_phase:
                            min_angle_in_phase = current_angle
                            if current_angle <= 170.0:
                                cv2.putText(annotated, "BENT KNEE DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                                worst_foul_frame = (current_angle, annotated.copy())

                        # 💡 [추적 종료 조건] '온몸이 수직이 되기까지' = 허리 축이 종골(디딤점)을 넘어가는 순간
                        if moving_right and waist_center[0] >= f_heel[0]:
                            tracking_phase = False
                        elif not moving_right and waist_center[0] <= f_heel[0]:
                            tracking_phase = False
                        elif stride_dist < (w * 0.03): # 예외: 발이 엇갈리기 시작할 때 강제 종료
                            tracking_phase = False
                            
                        # 수직 구간을 통과하며 추적이 끝났을 때, 기록된 최악의 파울 프레임을 최종 저장
                        if not tracking_phase and worst_foul_frame is not None:
                            photo_finish_frames.append(worst_foul_frame)

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

                    prev_stride_dist = stride_dist
                    prev_trend = trend
                    prev_waist_x = waist_center[0]

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 최종 리포트
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("📸 Rule 54 연속 추적 VAR 판독 리포트")
        
        st.subheader(f"🔴 Bent Knee (수직 통과 전 170도 이하 붕괴): 총 {len(photo_finish_frames)}회 적발")
        if len(photo_finish_frames) > 0:
            st.error("⚠️ 착지 시점부터 온몸이 수직이 되는 구간 내에서 '허리-무릎-종골' 각도가 170도 아래로 붕괴되었습니다. (가장 굽혀진 순간 박제)")
            for i in range(0, len(photo_finish_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(photo_finish_frames):
                        foul = photo_finish_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 #{i+j+1} (최저 각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 착지 후 온몸이 수직이 될 때까지 전 구간에서 무릎이 완벽하게 잠겨(Locked) 있었습니다.")
            
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
st.info("💡 **동작 원리:** 배측굴곡 상태에서 앞발의 **종골(뒤꿈치)**이 땅에 닿는 순간부터 추적을 시작하여, **허리 중심축**이 종골 위를 통과하는 수직 상태에 도달할 때까지 매 프레임 무릎 각도를 검사합니다.")
