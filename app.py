import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 1. 인터페이스 설정
st.set_page_config(page_title="Racewalking Foul Catcher", layout="wide")
st.title("🚨 엘리트 경보 Rule 54 파울 정밀 판독기")
st.markdown("##### 💡 쓸데없는 데이터는 버리고, '무릎 굽힘(Bent Knee)'과 '양발 체공(Loss of Contact)' 파울 순간만 낚아챕니다.")
st.write("---")

# 2. AI 분석 엔진
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

# 3. 영상 업로드
st.error("⚠️ **10초 이내의 영상**을 올려주세요. 파울이 발생한 찰나의 프레임을 박제하여 보여줍니다.")
video_file = st.file_uploader("경보 훈련 영상 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    # 파울 기록 저장소
    bent_knee_fouls = []     # (각도, 이미지)
    loss_of_contact_fouls = [] # (이미지)
    
    # 역학 추적 변수
    prev_stride_dist = 0
    prev_trend = 0
    global_ground_y = 0.0 # 지면(가장 낮은 발의 Y좌표) 추적
    
    # 쿨다운 (같은 파울을 연속으로 여러 장 찍지 않도록 방어)
    cooldown_bent_knee = 0
    cooldown_contact = 0
    
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("AI가 영상을 프레임 단위로 쪼개어 파울 순간을 수색 중입니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                h, w = frame.shape[:2]
                if w > 640: frame = cv2.resize(frame, (640, int(h * 640 / w)))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                # 쿨다운 감소
                if cooldown_bent_knee > 0: cooldown_bent_knee -= 1
                if cooldown_contact > 0: cooldown_contact -= 1
                
                if res.pose_landmarks:
                    person_detected = True
                    lm = res.pose_landmarks.landmark
                    
                    l_h = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                    r_h = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                    l_k = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                    r_k = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                    l_a = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                    r_a = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
                    nose_x = lm[mp_pose.PoseLandmark.NOSE].x
                    
                    # 💡 [핵심 1] 지면(Ground) 라인 실시간 추적
                    # 발목 중 가장 아래(Y값이 가장 큰) 좌표를 지속적으로 갱신
                    current_lowest_y = max(l_a[1], r_a[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y
                        
                    # 보폭(발목 사이의 X거리) 계산
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    
                    # 진행 방향 판별
                    hip_center_x = (l_h[0] + r_h[0]) / 2
                    facing_right = nose_x > hip_center_x 
                    
                    # 화면 표시용 복사본
                    annotated = img.copy()
                    mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # =========================================================
                    # 🚨 파울 추적 1: 앞다리 170도 이하 (Bent Knee)
                    # 원리: 보폭이 가장 넓어졌다가 좁아지기 시작하는 "착지(Peak)" 순간
                    # =========================================================
                    if trend < 0 and prev_trend > 0 and stride_dist > 0.05 and cooldown_bent_knee == 0:
                        # 앞다리가 무엇인지 확인
                        if facing_right:
                            leading_is_left = l_a[0] > r_a[0]
                        else:
                            leading_is_left = l_a[0] < r_a[0]
                            
                        # 앞다리 무릎 각도 계산
                        front_angle = calculate_angle(l_h, l_k, l_a) if leading_is_left else calculate_angle(r_h, r_k, r_a)
                        
                        # 170도 이하면 파울 처리 및 증거 수집!
                        if front_angle <= 170.0:
                            # 증거 사진에 빨간 글씨로 각도 새기기
                            cv2.putText(annotated, f"BENT KNEE FOUL: {front_angle:.1f} deg", (20, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                            bent_knee_fouls.append((front_angle, annotated.copy()))
                            cooldown_bent_knee = 15 # 다음 15프레임 동안은 중복 캡처 방지

                    # =========================================================
                    # 🚨 파울 추적 2: 양발 체공 (Loss of Contact)
                    # 원리: 두 다리가 교차하는(Valley) 0.42초의 순간, 두 발이 모두 지면 라인보다 한참 위에 있을 때
                    # =========================================================
                    if trend > 0 and prev_trend < 0 and stride_dist < 0.08 and cooldown_contact == 0:
                        # 현재 프레임에서 가장 낮은 발조차도 '지면 라인'보다 2% 이상 높이 떠 있다면 체공 파울!
                        if (global_ground_y - current_lowest_y) > 0.02: 
                            cv2.putText(annotated, "LOSS OF CONTACT FOUL", (20, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                            loss_of_contact_fouls.append(annotated.copy())
                            cooldown_contact = 15

                    prev_stride_dist = stride_dist
                    prev_trend = trend

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 파울 판독 결과 리포트
    if not person_detected:
        st.error("❌ 선수를 인식하지 못했습니다.")
    else:
        st.divider()
        st.header("🎯 Rule 54 판독 결과 리포트")
        
        # --- 1. 무릎 굽힘(Bent Knee) 결과 ---
        st.subheader(f"🔴 Bent Knee (170도 이하) 파울: 총 {len(bent_knee_fouls)}회 적발")
        if len(bent_knee_fouls) > 0:
            st.error("⚠️ 착지 순간 앞다리 무릎이 170도 이하로 붕괴되는 치명적인 파울이 포착되었습니다.")
            # 가로로 증거 사진 나열
            cols = st.columns(len(bent_knee_fouls))
            for idx, foul in enumerate(bent_knee_fouls):
                with cols[idx]:
                    st.image(foul[1], channels="RGB", caption=f"파울 {idx+1}: 무릎 {foul[0]:.1f}°")
        else:
            st.success("✅ Bent Knee 파울 없음: 착지 시 170도 이상의 무릎 신전을 훌륭하게 유지했습니다.")
            
        st.write("---")
        
        # --- 2. 체공(Loss of Contact) 결과 ---
        st.subheader(f"🟡 Loss of Contact (양발 체공) 파울: 총 {len(loss_of_contact_fouls)}회 적발")
        if len(loss_of_contact_fouls) > 0:
            st.warning("⚠️ 다리가 교차하는 찰나의 순간에 양발이 모두 지면에서 떨어지는 비행(Floating) 현상이 포착되었습니다.")
            cols2 = st.columns(len(loss_of_contact_fouls))
            for idx, img in enumerate(loss_of_contact_fouls):
                with cols2[idx]:
                    st.image(img, channels="RGB", caption=f"체공 파울 {idx+1} (교차 프레임)")
        else:
            st.success("✅ Loss of Contact 파울 없음: 교차 구간에서도 한 발이 완벽하게 지면을 지지하고 있습니다.")

st.write("---")
st.info("💡 **이 판독기는 국제 육상 연맹(World Athletics) Rule 54의 핵심인 '앞다리 신전'과 '교차 순간 체공'만을 수학적으로 필터링하여 증거 프레임을 추출합니다.**")
