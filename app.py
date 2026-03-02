import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Official VAR", layout="wide")
st.title("🚨 Rule 54 공식 VAR (0.08초 체공 & 신전 각도 판독)")
st.markdown("##### 💡 1. 두 발이 지면에서 '0.08초' 이상 떨어지면 즉각 실격 (Loss of Contact)")
st.markdown("##### 💡 2. 발이 지면에 닿는 순간, '골반-무릎-발목' 각도가 170도 이하로 굽혀지면 실격 (Bent Knee)")
st.write("---")

mp_pose = mp.solutions.pose
# 에러 방지를 위해 안정적인 모델 복잡도(1) 사용
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    # 골반(a) - 무릎(b) - 발목(c) 사이의 내각을 계산하여 신전(펴짐) 정도를 측정
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

st.error("⚠️ **10초 이내의 영상**을 올려주세요. AI가 0.08초 타이머와 무릎 각도를 동시에 스캔합니다.")
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
            
        # 💡 [핵심 룰 1] 체공 허용치: 0.08초 (심판 육안 한계치 초과)
        required_flight_frames = int(0.08 * fps)
        
        # 0.08초가 프레임 수로 0이 나오면 최소 2프레임으로 강제 지정 (초당 30프레임 기준 약 0.06초)
        if required_flight_frames < 2: 
            required_flight_frames = 2

        with st.spinner(f"🕵️‍♂️ 영상 분석 중... (0.08초 체공 파울 기준: {required_flight_frames} 프레임 이상 비행)"):
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
                    
                    # 💡 골반, 무릎 오금(Knee Joint), 발목 좌표
                    l_h = get_pt(lm[mp_pose.PoseLandmark.LEFT_HIP])
                    r_h = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                    l_k = get_pt(lm[mp_pose.PoseLandmark.LEFT_KNEE])
                    r_k = get_pt(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                    l_a = get_pt(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
                    r_a = get_pt(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                    nose_x = lm[mp_pose.PoseLandmark.NOSE].x * w
                    
                    # 진행 방향 판별
                    hip_center_x = (l_h[0] + r_h[0]) / 2
                    if prev_hip_x == 0: prev_hip_x = hip_center_x
                    moving_right = hip_center_x > prev_hip_x
                    
                    # 지면 기준선 업데이트
                    current_lowest_y = max(l_a[1], r_a[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y
                        
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    annotated = img.copy()

                    # =========================================================
                    # 🚨 [룰 1] 착지 순간: 골반-무릎-발목 신전 각도 측정
                    # =========================================================
                    # 발뒤꿈치가 지면에 닿는 순간 (보폭이 가장 넓은 정점)
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1):
                        leading_is_left = (l_a[0] > r_a[0]) if moving_right else (l_a[0] < r_a[0])

                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        
                        # 💡 골반-무릎-발목 사이의 내각 측정
                        front_angle = calculate_angle(front_hip, front_knee, front_ankle)
                        
                        if front_angle <= 170.0:
                            # 파울일 경우 뼈대 선명하게 묘사
                            line_thick = max(3, int(w / 200))
                            # 골반 -> 무릎 빨간선
                            cv2.line(annotated, tuple(front_hip), tuple(front_knee), (0, 0, 255), line_thick, cv2.LINE_AA)
                            # 무릎 -> 발목 빨간선
                            cv2.line(annotated, tuple(front_knee), tuple(front_ankle), (0, 0, 255), line_thick, cv2.LINE_AA)
                            # 무릎 관절 포인트 (노란색 점)
                            cv2.circle(annotated, tuple(front_knee), max(6, int(w/150)), (0, 255, 255), -1, cv2.LINE_AA)
                            
                            # 각도 텍스트 출력
                            cv2.putText(annotated, f"BENT KNEE: {front_angle:.1f} deg", (front_knee[0] + 15, front_knee[1]), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                            
                            photo_finish_frames.append((front_angle, annotated.copy()))

                    # =========================================================
                    # 🚨 [룰 2] 체공 시간 측정: 0.08초 초과 시 실격
                    # =========================================================
                    # 두 발 중 더 낮게 있는 발조차도 지면(global_ground_y)에서 떨어져 있다면 비행 중으로 간주
                    flight_gap = global_ground_y - current_lowest_y
                    
                    if flight_gap > (h * 0.02): # 화면 높이의 2% 이상 떴을 때 체공 프레임 누적
                        flight_frames_count += 1
                    else:
                        # 발이 땅에 닿는 순간, 그동안 누적된 체공 프레임이 0.08초(required_flight_frames) 이상인지 검사
                        if flight_frames_count >= required_flight_frames:
                            flight_time_sec = flight_frames_count / fps
                            
                            # 💡 0.08초가 넘은 명백한 체공 파울 박제
                            cv2.line(annotated, (0, int(global_ground_y)), (w, int(global_ground_y)), (0, 255, 0), 2)
                            cv2.putText(annotated, f"FLIGHT FOUL: {flight_time_sec:.3f} sec (>0.08s)", (20, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                            
                            flight_foul_frames.append(annotated.copy())
                            
                        # 체공 카운터 초기화
                        flight_frames_count = 0

                    prev_stride_dist = stride_dist
                    prev_trend = trend
                    prev_hip_x = hip_center_x

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 최종 판독 결과 출력
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("📸 Rule 54 공식 VAR 판독 리포트")
        
        # --- 1. 무릎 굽힘 (Bent Knee) ---
        st.subheader(f"🔴 Bent Knee (170도 이하): 총 {len(photo_finish_frames)}회 적발")
        if len(photo_finish_frames) > 0:
            st.error("⚠️ 착지 순간 전방 다리의 [골반-무릎-발목] 각도가 170도 이하로 굽혀져 수직 신전(Extension)에 실패했습니다.")
            for i in range(0, len(photo_finish_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(photo_finish_frames):
                        foul = photo_finish_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 #{i+j+1} (무릎 각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 착지 시 전방 다리가 수직선을 통과할 때까지 완벽한 신전 상태를 유지했습니다.")
            
        st.write("---")
        
        # --- 2. 체공 (Loss of Contact) ---
        st.subheader(f"🟡 Loss of Contact (0.08초 초과 체공): 총 {len(flight_foul_frames)}회 적발")
        if len(flight_foul_frames) > 0:
            st.warning("⚠️ 심판 판정의 엇갈림 구간(0.042~0.080초)을 지나, 즉각 실격 기준인 '0.08초'를 초과하여 두 발이 허공에 떠 있었습니다.")
            for i in range(0, len(flight_foul_frames), 3):
                cols2 = st.columns(3)
                for j in range(3):
                    if i + j < len(flight_foul_frames):
                        img = flight_foul_frames[i + j]
                        with cols2[j]:
                            st.image(img, channels="RGB", caption=f"0.08초 초과 비행 적발 #{i+j+1}")
        else:
            st.success("✅ Loss of Contact 통과: 체공 시간이 0.08초 이하(모호한 구간 혹은 지면 밀착)로 규칙을 준수했습니다.")

st.write("---")
st.info("💡 **판독 기준:** 체공 0.08초 초과 즉각 실격 / 골반-무릎오금-발목 170도 이하 신전 실패 시 실격")
