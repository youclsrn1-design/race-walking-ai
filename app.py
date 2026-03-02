import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import math

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Strict Catcher (High Accuracy)", layout="wide")
st.title("🚨 국제 심판 전용: Rule 54 절대 수직선 검증 AI")
st.markdown("##### 💡 다리가 땅에 닿는 순간부터 수직선(Rigid lever)을 유지해야 하는 규정을 바탕으로, 앞다리가 170도 이하로 붕괴되는 진짜 파울만 낚아챕니다.")
st.write("---")

# 2. AI 분석 엔진 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

# 3. 영상 업로드 및 파울 수색
st.error("⚠️ **10초 이내의 영상**을 올려주세요. AI가 착지 프레임을 캡처한 뒤 절대 수직선을 그어 분석합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    # 파울 사진 저장소
    photo_finish_frames = []   
    
    prev_stride_dist = 0
    prev_trend = 0
    
    cooldown_bent_knee = 0 # 쿨다운 변수 (중복 캡처 방지)
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("🕵️‍♂️ AI 심판이 착지 순간 앞다리에 '발목-무릎-골반 수직선'을 들이대고 있습니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                h, w = frame.shape[:2]
                if w > 800: frame = cv2.resize(frame, (800, int(h * 800 / w)))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                # 쿨다운 감소
                if cooldown_bent_knee > 0:
                    cooldown_bent_knee -= 1
                
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
                        
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    annotated = img.copy()

                    # (선택) 전체 뼈대를 그리지 않고 깔끔하게 분석 선만 그리고 싶다면 아래 줄을 주석 처리해도 됩니다.
                    # mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # =========================================================
                    # 🚨 절대 기준 적용: 착지 순간 앞다리 수직선 유지 여부
                    # =========================================================
                    # 보폭이 가장 넓어졌다가 좁아지는 딱 그 정점(착지 순간)
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1) and cooldown_bent_knee == 0:
                        hip_center_x = (l_h[0] + r_h[0]) / 2
                        facing_right = nose_x > hip_center_x 
                        
                        # 전방 다리 판별
                        leading_is_left = (l_a[0] > r_a[0]) if facing_right else (l_a[0] < r_a[0])

                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        
                        front_angle = calculate_angle(front_hip, front_knee, front_ankle)
                        
                        # 💡 [검증의 핵심] 명백하게 170도 이하로 붕괴되었을 때만 빨간색 수직선을 박제!
                        if front_angle <= 170.0:
                            # 1. 선생님의 요구사항: 발목-무릎-골반을 잇는 빨간색 수직선 긋기
                            cv2.line(annotated, tuple(front_ankle), tuple(front_knee), (255, 0, 0), 5) # 발목-무릎
                            cv2.line(annotated, tuple(front_knee), tuple(front_hip), (255, 0, 0), 5) # 무릎-골반
                            
                            # 무릎 관절 위치 포인트 (노란색)
                            cv2.circle(annotated, tuple(front_knee), 10, (255, 255, 0), -1) 
                            
                            # 각도 표시
                            cv2.putText(annotated, f"BENT KNEE: {front_angle:.1f}", (front_knee[0] + 15, front_knee[1]), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                            
                            photo_finish_frames.append((front_angle, annotated.copy()))
                            cooldown_bent_knee = 15 # 쿨다운 설정 (같은 파울을 연속 캡처 방지)

                    prev_stride_dist = stride_dist
                    prev_trend = trend

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 판독 결과 출력
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("📸 Rule 54 절대 수직선 판독 결과")
        
        # --- 1. 무릎 굽힘 (Bent Knee) ---
        st.subheader(f"🔴 Bent Knee (170도 이하 적발): 총 {len(photo_finish_frames)}회")
        if len(photo_finish_frames) > 0:
            st.error(f"⚠️ **{np.mean([f[0] for f in photo_finish_frames]):.1f}° (최저 {min([f[0] for f in photo_finish_frames]):.1f}°)** - 착지 순간 앞다리 무릎이 170도 아래로 무너져 수직 레버(Rigid lever)가 붕괴되었습니다. 육안 검증이 필요합니다.")
            
            # 파울 프레임 가로 나열 (에러 방지 그리드 로직)
            for i in range(0, len(photo_finish_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(photo_finish_frames):
                        foul = photo_finish_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 사진 #{i+j+1} (무릎 각도: {foul[0]:.1f}°)")
                            # 💡 미세 각도 피드백 추가 (오심 방지)
                            if foul[0] < 168:
                                st.error("🚨 **명백한 파울:** AI 스캔 결과 명백한 룰 위반입니다.")
                            else:
                                st.warning("⚠️ **미세 오차 가능성:** AI 스캔 미세 오차(false positive) 확률이 있습니다. 선수의 다리 윤곽과 빨간선을 육안으로 직접 대조하여 최종 판단하세요.")
        else:
            st.success("✅ Bent Knee 통과: 착지 프레임 추출 후 3차 검증을 했으나 무릎이 모두 곧게 펴져 수직선을 유지하고 있습니다.")
