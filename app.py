import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Rule 54 Racewalking VAR", layout="wide")
st.title("🚨 Rule 54 공식 VAR (경보 전용 연속 추적 엔진)")
st.markdown("##### 💡 1. 뻗은 앞발(종골)이 허리 중심축보다 '앞에 있는 모든 순간'을 연속 감시합니다.")
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

st.error("⚠️ **10초 이내의 경보 영상**을 올려주세요. 앞발이 몸통을 지나갈 때까지 무릎 붕괴를 추적합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    photo_finish_frames = []   
    flight_foul_frames = []
    
    global_ground_y = 0.0
    flight_frames_count = 0 
    
    # 💡 [경보 전용] 걸음(Step) 추적 변수
    prev_front_leg = None
    worst_foul_frame = None
    min_angle_in_step = 360.0
    
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
            
        required_flight_frames = int(0.08 * fps)
        if required_flight_frames < 2: required_flight_frames = 2

        with st.spinner(f"🕵️‍♂️ 영상 분석 중... (달리기 로직 폐기, 걷기 기반 앞발 연속 추적 중)"):
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
                    
                    # 좌표 가져오기 (종골, 무릎, 골반, 코)
                    l_h = get_pt(lm[mp_pose.PoseLandmark.LEFT_HIP])
                    r_h = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                    l_k = get_pt(lm[mp_pose.PoseLandmark.LEFT_KNEE])
                    r_k = get_pt(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                    l_heel = get_pt(lm[mp_pose.PoseLandmark.LEFT_HEEL])
                    r_heel = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HEEL])
                    nose_x = lm[mp_pose.PoseLandmark.NOSE].x * w
                    
                    # 허리 중심축 (골반 사이 정중앙)
                    waist_center = [int((l_h[0] + r_h[0]) / 2), int((l_h[1] + r_h[1]) / 2)]
                    
                    # 진행 방향 (코가 허리보다 어느 쪽에 있는가로 명확히 판별)
                    moving_right = nose_x > waist_center[0]
                    
                    # 현재 "앞발(Front Leg)"이 왼발인지 오른발인지 판별
                    if moving_right:
                        front_is_left = l_heel[0] > r_heel[0]
                    else:
                        front_is_left = l_heel[0] < r_heel[0]
                        
                    current_front_leg = 'L' if front_is_left else 'R'

                    # 지면 기준선 업데이트
                    current_lowest_y = max(l_heel[1], r_heel[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y
                        
                    annotated = img.copy()

                    # =========================================================
                    # 🚨 [1단계] 걸음(Step) 전환 감지 및 초기화
                    # =========================================================
                    # 앞발이 왼발->오른발(또는 반대)로 바뀌었다면, 새로운 걸음이 시작된 것!
                    if current_front_leg != prev_front_leg:
                        # 이전 걸음에서 파울이 있었다면 최종 바구니에 저장
                        if worst_foul_frame is not None:
                            photo_finish_frames.append(worst_foul_frame)
                        
                        # 새로운 걸음을 위해 추적기 초기화
                        worst_foul_frame = None
                        min_angle_in_step = 360.0
                        prev_front_leg = current_front_leg

                    # =========================================================
                    # 🚨 [2단계] "앞발이 허리보다 앞에 있는가?" (연속 감시 구간)
                    # =========================================================
                    f_heel = l_heel if front_is_left else r_heel
                    f_knee = l_k if front_is_left else r_k
                    
                    is_in_support_phase = False
                    
                    # 💡 달리기식 착지 정점 로직 폐기!
                    # 종골(뒤꿈치)이 허리 중심축보다 '앞'에만 있다면 무조건 판독 구간!
                    if moving_right and f_heel[0] > waist_center[0]:
                        is_in_support_phase = True
                    elif not moving_right and f_heel[0] < waist_center[0]:
                        is_in_support_phase = True

                    if is_in_support_phase:
                        # 허리중심 - 무릎중심 - 종골 각도 계산
                        current_angle = calculate_angle(waist_center, f_knee, f_heel)
                        
                        # 시각화 묘사 (앞발이 앞에 있을 때 선 긋기)
                        line_thick = max(4, int(w / 180))
                        cv2.line(annotated, tuple(waist_center), tuple(f_knee), (0, 0, 255), line_thick, cv2.LINE_AA)
                        cv2.line(annotated, tuple(f_knee), tuple(f_heel), (0, 0, 255), line_thick, cv2.LINE_AA)
                        
                        cv2.circle(annotated, tuple(waist_center), 8, (255, 0, 255), -1) # 허리 축 (보라)
                        cv2.circle(annotated, tuple(f_knee), 8, (0, 255, 255), -1)      # 무릎 축 (노랑)
                        cv2.circle(annotated, tuple(f_heel), 8, (0, 255, 0), -1)        # 종골 디딤점 (초록)
                        
                        cv2.putText(annotated, f"ANGLE: {current_angle:.1f}", (f_knee[0] + 20, f_knee[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w/900), (0, 255, 255), 3)

                        # 170도 이하로 무너졌다면 파울 기록 갱신 (가장 심하게 꺾인 프레임 하나만 저장)
                        if current_angle <= 170.0:
                            if current_angle < min_angle_in_step:
                                min_angle_in_step = current_angle
                                cv2.putText(annotated, "BENT KNEE DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
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

            # 영상이 끝났을 때 마지막 걸음의 파울도 마저 담아주기
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
        st.header("📸 Rule 54 연속 추적 VAR 판독 리포트 (걷기 전용)")
        
        st.subheader(f"🔴 Bent Knee (수직 통과 전 170도 이하 붕괴): 총 {len(photo_finish_frames)}회 적발")
        if len(photo_finish_frames) > 0:
            st.error("⚠️ 앞발(종골)이 허리보다 앞에 있는 모든 순간을 추적한 결과, 무릎이 170도 아래로 붕괴된 스텝들입니다. (해당 스텝 중 최저 각도 박제)")
            for i in range(0, len(photo_finish_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(photo_finish_frames):
                        foul = photo_finish_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 #{i+j+1} (최저 각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 앞발이 몸통을 통과하는 전 구간에서 무릎이 완벽하게 펴져 있었습니다.")
            
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
st.info("💡 **동작 원리 (걷기 역학 적용):** 달리기 식의 '최대 보폭점' 계산을 폐기했습니다. 앞발의 **종골(뒤꿈치)**이 **허리 중심축**보다 앞에 나와 있는 찰나의 모든 프레임을 1프레임 단위로 감시하여 가장 심하게 꺾인 파울 장면을 캡처합니다.")
