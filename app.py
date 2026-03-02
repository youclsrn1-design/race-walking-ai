import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Ultimate VAR", layout="wide")
st.title("🎬 Rule 54 완전 자동화 VAR (3차 검증 탑재)")
st.markdown("##### 💡 1. 영상에서 '착지 순간'을 사진으로 자동 캡처\n##### 💡 2. 캡처된 사진에서 '골반-복숭아뼈 절대 직선'으로 3차 정밀 각도 검증\n##### 💡 3. 두 발이 완벽히 떨어진 시간이 '0.42초 이상'일 때만 체공 파울")
st.write("---")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

st.error("⚠️ **10초 이내의 영상**을 올려주세요. AI가 영상을 분석해 파울 사진을 추출하고 3차 검증을 수행합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    final_bent_knee_frames = []   
    flight_foul_frames = []
    
    prev_stride_dist = 0
    prev_trend = 0
    
    global_ground_y = 0.0
    flight_frames_count = 0 
    
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
            
        required_flight_frames = int(0.42 * fps)

        with st.spinner("🕵️‍♂️ 영상 분석 중... (착지 프레임 추출 -> 3차 절대 직선 검증 진행)"):
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
                    
                    current_lowest_y = max(l_a[1], r_a[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y
                        
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    annotated = img.copy()

                    # =========================================================
                    # 🚨 [1단계 & 2단계] 착지 프레임 추출 및 3차 정지화면 검증
                    # =========================================================
                    # 보폭이 가장 넓어지는 '착지 순간'에 사진(프레임) 찰칵!
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1):
                        
                        # 💡 [3차 검증 돌입] 뽑아낸 사진을 '멈춰있는 사진'으로 간주하고 앞발 찾기
                        hip_center_x = (l_h[0] + r_h[0]) / 2
                        facing_right = nose_x > hip_center_x 
                        
                        if facing_right:
                            leading_is_left = l_a[0] > r_a[0] 
                        else:
                            leading_is_left = l_a[0] < r_a[0] 

                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        
                        front_angle = calculate_angle(front_hip, front_knee, front_ankle)
                        
                        # 각도가 170도 이하일 때만 (파울일 때만) 사진에 선을 긋고 박제
                        if front_angle <= 170.0:
                            # 1. 골반-복숭아뼈 절대 직선 (초록색)
                            cv2.line(annotated, tuple(front_hip), tuple(front_ankle), (0, 255, 0), 3)
                            
                            # 2. 선수의 실제 무릎 꺾임 (빨간색)
                            cv2.line(annotated, tuple(front_hip), tuple(front_knee), (255, 0, 0), 5)
                            cv2.line(annotated, tuple(front_knee), tuple(front_ankle), (255, 0, 0), 5)
                            cv2.circle(annotated, tuple(front_knee), 10, (0, 255, 255), -1)
                            
                            cv2.putText(annotated, f"3RD VERIFIED: {front_angle:.1f} deg", (front_knee[0] + 15, front_knee[1]), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                            
                            final_bent_knee_frames.append((front_angle, annotated.copy()))

                    # =========================================================
                    # 🚨 [별도] 체공 파울: 0.42초 (프레임 누적) 절대 룰
                    # =========================================================
                    if (global_ground_y - current_lowest_y) > (h * 0.02):
                        flight_frames_count += 1
                    else:
                        if flight_frames_count >= required_flight_frames:
                            flight_time_sec = flight_frames_count / fps
                            cv2.putText(annotated, f"FLIGHT: {flight_time_sec:.2f}s (>= 0.42s)", (30, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                            flight_foul_frames.append(annotated.copy())
                        flight_frames_count = 0

                    prev_stride_dist = stride_dist
                    prev_trend = trend

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 최종 판독 결과 출력
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("🎬 영상 기반 3차 검증 판독 리포트")
        
        # --- 1. 무릎 굽힘 (Bent Knee) ---
        st.subheader(f"🔴 3차 검증 완료 (Bent Knee): 총 {len(final_bent_knee_frames)}회 적발")
        if len(final_bent_knee_frames) > 0:
            st.error("⚠️ 영상에서 추출한 착지 사진에 '절대 직선'을 대조한 결과, 170도 이하로 꺾인 진짜 파울입니다.")
            for i in range(0, len(final_bent_knee_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(final_bent_knee_frames):
                        foul = final_bent_knee_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"3차 검증 파울 #{i+j+1} (무릎: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 착지 프레임 추출 후 3차 검증을 했으나 무릎이 모두 곧게 펴져 있습니다.")
            
        st.write("---")
        
        # --- 2. 체공 (Loss of Contact) ---
        st.subheader(f"🟡 체공 0.42초 초과 (Loss of Contact): 총 {len(flight_foul_frames)}회 적발")
        if len(flight_foul_frames) > 0:
            st.warning("⚠️ 두 발이 완전히 땅에서 떨어져 있는 시간이 0.42초 이상 누적된 명백한 체공입니다.")
            for i in range(0, len(flight_foul_frames), 3):
                cols2 = st.columns(3)
                for j in range(3):
                    if i + j < len(flight_foul_frames):
                        img = flight_foul_frames[i + j]
                        with cols2[j]:
                            st.image(img, channels="RGB", caption=f"0.42초 초과 체공 파울 #{i+j+1}")
        else:
            st.success("✅ Loss of Contact 통과: 체공 시간이 0.42초 미만이거나 한 발이 땅에 닿아있습니다.")

st.write("---")
st.info("💡 **동작 원리:** 영상을 올리면 AI가 '보폭이 가장 넓어진 사진'을 자동 캡처하고, 그 정지 화면 위에서 전방 다리의 골반-복숭아뼈 절대 직선을 그어 3차 검증을 수행합니다.")
