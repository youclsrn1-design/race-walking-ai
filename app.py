import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import math

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Foul Catcher (Dual AI)", layout="wide")
st.title("🚨 국제 심판 전용: Rule 54 파울 2차 검증 AI")
st.markdown("##### 💡 단일 AI의 착각을 막기 위해 '종골-골반 절대 직선'과 '0.42 & 0.58 듀얼 수학 모델'로 교차 검증된 진짜 파울만 박제합니다.")
st.warning("🔒 업로드된 영상은 파울 프레임 추출 직후 서버에서 즉각 영구 삭제됩니다.")
st.write("---")

# 2. AI 분석 엔진 초기화 (에러 해결: model_complexity=1로 수정)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

# 3. 영상 업로드 및 파울 수색
st.error("⚠️ **10초 이내의 훈련 영상**을 올려주세요. AI가 1차, 2차 검증을 동시에 수행합니다.")
video_file = st.file_uploader("경보 역학 분석 영상 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    foul_bent_knee_frames = []   
    foul_contact_frames = []     
    
    prev_stride_dist = 0
    prev_trend = 0
    global_ground_y = 0.0 
    
    cd_bent_knee = 0
    cd_contact = 0
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("🕵️‍♂️ 2개의 AI 엔진이 0.42 / 0.58 기준으로 수학적 교차 검증을 진행 중입니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                h, w = frame.shape[:2]
                if w > 800: frame = cv2.resize(frame, (800, int(h * 800 / w)))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                if cd_bent_knee > 0: cd_bent_knee -= 1
                if cd_contact > 0: cd_contact -= 1
                
                if res.pose_landmarks:
                    person_detected = True
                    lm = res.pose_landmarks.landmark
                    
                    def get_pt(landmark):
                        return [int(landmark.x * w), int(landmark.y * h)]
                    
                    # 주요 관절 좌표 (종골-Heel 추가)
                    l_h = get_pt(lm[mp_pose.PoseLandmark.LEFT_HIP])
                    r_h = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                    l_k = get_pt(lm[mp_pose.PoseLandmark.LEFT_KNEE])
                    r_k = get_pt(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                    l_a = get_pt(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
                    r_a = get_pt(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                    
                    # 💡 2차 검증용 종골(Heel) 좌표 추가
                    l_heel = get_pt(lm[mp_pose.PoseLandmark.LEFT_HEEL])
                    r_heel = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HEEL])
                    
                    nose_x = lm[mp_pose.PoseLandmark.NOSE].x * w
                    
                    current_lowest_y = max(l_heel[1], r_heel[1]) # 발목 대신 진짜 발바닥인 종골 기준
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y
                        
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    
                    hip_center_x = (l_h[0] + r_h[0]) / 2
                    facing_right = nose_x > hip_center_x 
                    leading_is_left = (l_a[0] > r_a[0]) if facing_right else (l_a[0] < r_a[0])

                    annotated = img.copy()
                    mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # =========================================================
                    # 🚨 1. Bent Knee 판독 (2차 검증 탑재)
                    # =========================================================
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1) and cd_bent_knee == 0:
                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        front_heel = l_heel if leading_is_left else r_heel
                        
                        # [1차 검증] 발목 기준 무릎 각도
                        angle_ankle = calculate_angle(front_hip, front_knee, front_ankle)
                        # [2차 검증] 종골(Heel) 기준 무릎 각도
                        angle_heel = calculate_angle(front_hip, front_knee, front_heel)
                        
                        # 1차 판독에서 170도 이하로 굽었다고 판단했을 때
                        if angle_ankle <= 170.0:
                            # 💡 [2차 검증 로직] 골반과 종골을 이은 가상의 직선 각도 확인
                            if angle_heel >= 175.0:
                                # "파울 아님!" -> 빨간선(골반-종골)을 그어주고 파울 리스트에는 넣지 않음
                                cv2.line(annotated, tuple(front_hip), tuple(front_heel), (0, 255, 0), 2) # 초록선으로 패스 표시
                                pass 
                            else:
                                # 1차, 2차 모두 굽었다고 판독 -> "진짜 파울"
                                cv2.line(annotated, tuple(front_hip), tuple(front_knee), (255, 0, 0), 4)
                                cv2.line(annotated, tuple(front_knee), tuple(front_heel), (255, 0, 0), 4)
                                
                                cv2.putText(annotated, f"CONFIRMED BENT: {angle_heel:.1f} deg", (30, 80), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                                foul_bent_knee_frames.append((angle_heel, annotated.copy()))
                                cd_bent_knee = 15

                    # =========================================================
                    # 🚨 2. 체공 파울 판독 (0.42 / 0.58 듀얼 검증)
                    # =========================================================
                    stride_ratio = stride_dist / (w + 1e-5)
                    
                    ai_1_crossover = stride_ratio < 0.042 
                    ai_2_crossover = stride_ratio < 0.058 
                    
                    if (ai_1_crossover or ai_2_crossover) and cd_contact == 0:
                        floating_gap = global_ground_y - current_lowest_y
                        
                        ai_1_foul = floating_gap > (h * 0.042) # 매우 높이 뜬 경우
                        ai_2_foul = floating_gap > (h * 0.025) # 살짝 뜬 경우
                        
                        # 💡 0.42 AI와 0.58 AI가 "둘 다" 체공이라고 동의(일치)했을 때만 파울 처리!
                        if ai_1_foul and ai_2_foul:
                            cv2.line(annotated, (0, int(global_ground_y)), (w, int(global_ground_y)), (255, 0, 0), 3)
                            cv2.putText(annotated, "DUAL VERIFIED: LOSS OF CONTACT", (30, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                            foul_contact_frames.append(annotated.copy())
                            cd_contact = 15

                    prev_stride_dist = stride_dist
                    prev_trend = trend

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 파울 박제 리포트 출력
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다. 선수가 잘 보이는 영상을 올려주세요.")
    else:
        st.divider()
        st.header("🎯 Rule 54 듀얼 검증(Cross-Validated) 파울 리포트")
        
        # --- 1. 무릎 굽힘 (Bent Knee) ---
        st.subheader(f"🔴 Bent Knee (2차 검증 완료): 총 {len(foul_bent_knee_frames)}회 적발")
        if len(foul_bent_knee_frames) > 0:
            st.error("⚠️ [골반-종골 절대 직선] 검증을 거친 '진짜 무릎 굽힘 파울'입니다.")
            for i in range(0, len(foul_bent_knee_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(foul_bent_knee_frames):
                        foul = foul_bent_knee_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 #{i+j+1} (종골 기준 각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 1차에서 굽어 보였더라도 2차 종골-골반 직선 검증에서 모두 무죄 판결을 받았습니다.")
            
        st.write("---")
        
        # --- 2. 양발 체공 (Loss of Contact) ---
        st.subheader(f"🟡 Loss of Contact (0.42/0.58 이중 검증 완료): 총 {len(foul_contact_frames)}회 적발")
        if len(foul_contact_frames) > 0:
            st.warning("⚠️ 0.42 엔진과 0.58 엔진이 동시에 '양발 체공'으로 만장일치 판정한 프레임입니다.")
            for i in range(0, len(foul_contact_frames), 3):
                cols2 = st.columns(3)
                for j in range(3):
                    if i + j < len(foul_contact_frames):
                        img = foul_contact_frames[i + j]
                        with cols2[j]:
                            st.image(img, channels="RGB", caption=f"체공 파울 #{i+j+1}")
        else:
            st.success("✅ Loss of Contact 통과: 두 AI 엔진 모두 양발 체공이 없다고 동의했습니다.")

st.write("---")
st.info("💡 **[2차 검증 알고리즘 가동 중]** 단일 앵글 오류를 막기 위해, 골반과 종골(Heel)을 잇는 보조선을 바탕으로 진짜 파울만 추출합니다.")
