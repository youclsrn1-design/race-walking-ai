import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Strict Catcher", layout="wide")
st.title("🚨 국제 심판 전용: Rule 54 절대 직선 검증 AI")
st.markdown("##### 💡 골반과 복숭아뼈를 잇는 '절대 직선'을 그어 무릎이 170도 이하로 무너지는 진짜 파울만 낚아챕니다.")
st.write("---")

# 2. AI 분석 엔진 초기화 (에러 원천 차단을 위해 Complexity 1 고정)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

# 3. 영상 업로드 및 파울 수색
st.error("⚠️ **10초 이내의 영상**을 올려주세요. AI가 착지 프레임을 캡처한 뒤 절대 직선을 그어 분석합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    # 💡 [에러 해결] 바구니 이름 통일 및 초기화
    foul_bent_knee_frames = []   
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
            
        required_flight_frames = int(0.42 * fps)

        # 💡 빨간선/노란점 그리는 내부 함수
        def draw_strict_analytical_lines(img, hip, knee, ankle, angle):
            h, w, _ = img.shape
            
            # 굵기 설정 (화면 크기 비례)
            line_thick = max(4, int(w / 200))
            circle_rad = max(10, int(w / 100))
            font_scale = max(1.0, w / 700)
            font_thick = max(2, int(w / 400))

            # 1. 골반-무릎 빨간선
            cv2.line(img, tuple(hip), tuple(knee), (0, 0, 255), line_thick, cv2.LINE_AA)
            # 2. 무릎-발목 빨간선
            cv2.line(img, tuple(knee), tuple(ankle), (0, 0, 255), line_thick, cv2.LINE_AA)
            # 3. 무릎 위치 노란색 점
            cv2.circle(img, tuple(knee), circle_rad, (0, 255, 255), -1, cv2.LINE_AA)
            # 4. 각도 표시
            cv2.putText(img, f"ANGLE: {angle:.1f}", (knee[0] + 20, knee[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thick, cv2.LINE_AA)

        with st.spinner("🕵️‍♂️ 영상 분석 중... (착지 프레임 추출 -> 골반-복숭아뼈 빨간선 검증)"):
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
                    
                    # 방향 추적
                    hip_center_x = (l_h[0] + r_h[0]) / 2
                    if prev_hip_x == 0: prev_hip_x = hip_center_x
                    moving_right = hip_center_x > prev_hip_x
                    
                    # 지면 추적
                    current_lowest_y = max(l_a[1], r_a[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y
                        
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    annotated = img.copy()

                    # =========================================================
                    # 🚨 1단계: 착지 추출 & 빨간선 긋기
                    # =========================================================
                    # 보폭이 넓어졌다가 줄어들기 시작하는 착지 정점
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1):
                        leading_is_left = (l_a[0] > r_a[0]) if moving_right else (l_a[0] < r_a[0])

                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        
                        front_angle = calculate_angle(front_hip, front_knee, front_ankle)
                        
                        if front_angle <= 170.0:
                            # 💡 파울로 판정된 사진에 빨간선 긋기
                            draw_strict_analytical_lines(annotated, front_hip, front_knee, front_ankle, front_angle)
                            foul_bent_knee_frames.append((front_angle, annotated.copy()))

                    # =========================================================
                    # 🚨 2단계: 체공 파울 (0.42초 이상 떴을 때만)
                    # =========================================================
                    if (global_ground_y - current_lowest_y) > (h * 0.02):
                        flight_frames_count += 1
                    else:
                        if flight_frames_count >= required_flight_frames:
                            flight_time_sec = flight_frames_count / fps
                            cv2.putText(annotated, f"FLIGHT: {flight_time_sec:.2f} sec", (30, 100), 
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

    # 4. 최종 리포트 출력
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("📸 Rule 54 절대 직선 판독 결과")
        
        # --- 1. 무릎 굽힘 (Bent Knee) ---
        st.subheader(f"🔴 Bent Knee (170도 이하 적발): 총 {len(foul_bent_knee_frames)}회")
        if len(foul_bent_knee_frames) > 0:
            st.error(f"⚠️ 착지 순간 앞다리의 무릎(노란 점)이 170도 이하로 무너지는 것이 감지되었습니다.")
            
            for i in range(0, len(foul_bent_knee_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(foul_bent_knee_frames):
                        foul = foul_bent_knee_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 사진 #{i+j+1} (무릎 각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ **Bent Knee 통과:** 추출된 착지 프레임들의 무릎 각도가 모두 170도 이상입니다.")
            
        st.write("---")
        
        # --- 2. 체공 (Loss of Contact) ---
        st.subheader(f"🟡 Loss of Contact (0.42초 이상 체공): 총 {len(flight_foul_frames)}회 적발")
        if len(flight_foul_frames) > 0:
            st.warning("⚠️ 두 발이 완전히 땅에서 떨어져 있는 시간이 0.42초 이상 누적된 명백한 도약 파울입니다.")
            
            for i in range(0, len(flight_foul_frames), 3):
                cols2 = st.columns(3)
                for j in range(3):
                    if i + j < len(flight_foul_frames):
                        img = flight_foul_frames[i + j]
                        with cols2[j]:
                            st.image(img, channels="RGB", caption=f"0.42초 이상 체공 적발 #{i+j+1}")
        else:
            st.success("✅ **Loss of Contact 통과:** 두 발이 떨어져 있는 시간이 0.42초 미만이거나 한 발이 땅에 닿아있습니다.")

st.write("---")
st.info("💡 **알고리즘 룰 강제:** 골반-복숭아뼈 선을 긋고, 0.42초 이상 누적 체공될 경우에만 파울로 적발합니다.")
