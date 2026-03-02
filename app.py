import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Rule 54 Strict Catcher", layout="wide")
st.title("📸 Rule 54 초정밀 사진 판독기 (Strict Mode)")
st.markdown("##### 💡 1. 골반-복숭아뼈 절대 직선 기준 무릎 각도 판독\n##### 💡 2. 두 발이 완벽히 떨어진 시간이 '0.42초 이상'일 때만 체공 파울 인정")
st.warning("🔒 분석 완료 후 영상은 즉각 파기됩니다.")
st.write("---")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

st.error("⚠️ **10초 이내의 훈련 영상**을 올려주세요.")
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
    flight_frames_count = 0 # 공중에 떠 있는 프레임 수 누적
    
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0 # FPS를 못 가져오면 기본 30으로 설정
            
        # 0.42초가 되기 위해 필요한 프레임 수 계산
        required_flight_frames = int(0.42 * fps)

        with st.spinner(f"📸 분석 중... (체공 파울 기준: 두 발이 {required_flight_frames} 프레임 이상 뜰 경우)"):
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
                    
                    # 이동 방향 판별 (골반 중심 X좌표 추적)
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
                    # 🚨 1. 착지 순간: 골반-복숭아뼈 직선 기반 무릎 각도 측정
                    # =========================================================
                    # 보폭이 넓어졌다가 줄어들기 시작하는 딱 그 정점(착지 순간)
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1):
                        leading_is_left = (l_a[0] > r_a[0]) if moving_right else (l_a[0] < r_a[0])

                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        
                        front_angle = calculate_angle(front_hip, front_knee, front_ankle)
                        
                        # 💡 요구사항: 골반과 복숭아뼈를 직선으로 잇기
                        cv2.line(annotated, tuple(front_hip), tuple(front_ankle), (0, 255, 0), 2) # 초록색 절대 직선
                        
                        # 무릎이 그 직선 상에 놓여있는지 확인하기 위한 꺾임선(빨간선)
                        cv2.line(annotated, tuple(front_hip), tuple(front_knee), (255, 0, 0), 4)
                        cv2.line(annotated, tuple(front_knee), tuple(front_ankle), (255, 0, 0), 4)
                        cv2.circle(annotated, tuple(front_knee), 8, (0, 255, 255), -1) # 무릎 포인트 (노란색)
                        
                        # 각도 표시
                        color = (255, 0, 0) if front_angle <= 170.0 else (0, 255, 0)
                        cv2.putText(annotated, f"KNEE ANGLE: {front_angle:.1f}", (front_knee[0] + 15, front_knee[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                        
                        if front_angle <= 170.0:
                            photo_finish_frames.append((front_angle, annotated.copy()))

                    # =========================================================
                    # 🚨 2. 체공 파울: 두 발이 떨어져 있는 시간이 '0.42초 이상'일 때만
                    # =========================================================
                    # 가장 낮은 발조차도 땅(global_ground_y)에서 떨어져 있다면 (오차범위 2% 허용)
                    if (global_ground_y - current_lowest_y) > (h * 0.02):
                        flight_frames_count += 1
                    else:
                        # 발이 땅에 닿는 순간, 그동안 누적된 체공 시간이 0.42초(요구 프레임) 이상이었는지 검사
                        if flight_frames_count >= required_flight_frames:
                            flight_time_sec = flight_frames_count / fps
                            cv2.putText(annotated, f"FLIGHT FOUL: {flight_time_sec:.2f} sec", (30, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                            flight_foul_frames.append(annotated.copy())
                        # 초기화 (한 발이라도 땅에 닿으면 체공 카운트 리셋)
                        flight_frames_count = 0

                    prev_stride_dist = stride_dist
                    prev_trend = trend
                    prev_hip_x = hip_center_x

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 판독 결과 출력
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("📸 Rule 54 판독 결과")
        
        # --- 1. 무릎 굽힘 (Bent Knee) ---
        st.subheader(f"🔴 Bent Knee (170도 이하): 총 {len(photo_finish_frames)}회 적발")
        if len(photo_finish_frames) > 0:
            st.error("⚠️ 착지 순간 앞다리의 골반-복숭아뼈 절대 직선(초록선)에서 무릎(빨간선)이 170도 이하로 벗어났습니다.")
            
            for i in range(0, len(photo_finish_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(photo_finish_frames):
                        foul = photo_finish_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 #{i+j+1} (무릎 각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 착지 프레임에서 무릎이 절대 직선 상에 훌륭하게 정렬되어 있습니다.")
            
        st.write("---")
        
        # --- 2. 체공 (Loss of Contact) ---
        st.subheader(f"🟡 Loss of Contact (0.42초 이상 체공): 총 {len(flight_foul_frames)}회 적발")
        if len(flight_foul_frames) > 0:
            st.warning(f"⚠️ 두 발이 완전히 땅에서 떨어져 있는 시간이 0.42초를 초과한 명백한 도약 파울입니다.")
            
            for i in range(0, len(flight_foul_frames), 3):
                cols2 = st.columns(3)
                for j in range(3):
                    if i + j < len(flight_foul_frames):
                        img = flight_foul_frames[i + j]
                        with cols2[j]:
                            st.image(img, channels="RGB", caption=f"0.42초 이상 체공 적발 #{i+j+1}")
        else:
            st.success("✅ Loss of Contact 통과: 두 발이 떨어져 있는 시간이 0.42초 미만이거나, 항상 한 발이 땅에 닿아있습니다.")

st.write("---")
st.info("💡 **알고리즘 룰 강제:** 골반-복숭아뼈 직선을 기준선으로 삼고, 0.42초 이상 누적 체공될 경우에만 파울로 적발합니다.")
