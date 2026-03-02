import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Photo Finish", layout="wide")
st.title("📸 Rule 54 사진 판독기 (정밀 각도 측정)")
st.markdown("##### 💡 다리가 가장 넓게 벌어진 '착지 순간'의 사진을 캡처한 뒤, 앞다리에 2차 검증(골반-발목 직선)과 무릎 각도를 정밀하게 그려냅니다.")
st.warning("🔒 분석 완료 후 영상은 즉각 파기됩니다.")
st.write("---")

# 2. AI 분석 엔진
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

# 3. 영상 업로드
st.error("⚠️ **10초 이내의 훈련 영상**을 올려주세요. 착지 프레임을 추출하여 사진 판독을 진행합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    photo_finish_frames = []   
    
    prev_stride_dist = 0
    prev_trend = 0
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("📸 AI가 가장 완벽한 착지 프레임을 찾아 사진 판독을 진행 중입니다..."):
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
                        
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    
                    # 💡 보폭이 가장 넓어졌다가 좁아지는 딱 그 순간(Peak) = 완벽한 착지 프레임!
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1):
                        hip_center_x = (l_h[0] + r_h[0]) / 2
                        facing_right = nose_x > hip_center_x 
                        
                        # 사진(정지화면) 상태에서는 X좌표만으로 앞다리 구분이 100% 정확함
                        leading_is_left = (l_a[0] > r_a[0]) if facing_right else (l_a[0] < r_a[0])

                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        
                        front_angle = calculate_angle(front_hip, front_knee, front_ankle)
                        
                        annotated = img.copy()
                        
                        # =========================================================
                        # 📸 선생님이 원하시는 '2번째 사진 방식'의 정밀 선 긋기
                        # =========================================================
                        
                        # 1. 이상적인 180도 '절대 직선' 긋기 (골반에서 발목까지 다이렉트, 초록색 점선 느낌)
                        # OpenCV에서 점선 그리기가 까다로워 얇은 초록선으로 대체
                        cv2.line(annotated, tuple(front_hip), tuple(front_ankle), (0, 255, 0), 2)
                        
                        # 2. 선수의 실제 다리 뼈대 꺾임 긋기 (빨간색 굵은 선)
                        cv2.line(annotated, tuple(front_hip), tuple(front_knee), (255, 0, 0), 5) # 골반 -> 무릎
                        cv2.line(annotated, tuple(front_knee), tuple(front_ankle), (255, 0, 0), 5) # 무릎 -> 발목
                        
                        # 3. 무릎 관절 위치에 포인트 원 그리기
                        cv2.circle(annotated, tuple(front_knee), 8, (255, 255, 0), -1) 
                        
                        # 4. 각도 텍스트 출력
                        color = (255, 0, 0) if front_angle <= 170.0 else (0, 255, 0)
                        status = "FOUL (BENT)" if front_angle <= 170.0 else "PASS"
                        
                        cv2.putText(annotated, f"ANGLE: {front_angle:.1f} deg", (front_knee[0] + 20, front_knee[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                        cv2.putText(annotated, f"[{status}]", (front_knee[0] + 20, front_knee[1] + 35), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                        
                        # 170도 이하인 파울 장면만 수집
                        if front_angle <= 170.0:
                            photo_finish_frames.append((front_angle, annotated.copy()))

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
        st.header("📸 정지화면(Photo Finish) 무릎 각도 판독 결과")
        
        st.subheader(f"🔴 Bent Knee 파울 적발 사진: 총 {len(photo_finish_frames)}장")
        if len(photo_finish_frames) > 0:
            st.error("⚠️ 착지 프레임을 캡처한 뒤 정밀 각도를 재어본 결과, 파울이 확인되었습니다.")
            st.markdown("- **초록색 얇은 선:** 규정상 요구되는 골반-발목 간 '이상적인 180도 절대 직선'")
            st.markdown("- **빨간색 굵은 선:** 선수의 실제 무릎 꺾임 각도")
            
            for i in range(0, len(photo_finish_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(photo_finish_frames):
                        foul = photo_finish_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 사진 #{i+j+1} (무릎: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 착지 프레임 분석 결과 모두 170도를 넘겼습니다.")

st.write("---")
st.info("💡 실시간 추적의 오류를 없애고, **가장 보폭이 넓어진 착지 프레임(정지화면)**을 캡처한 후 그 위에서 각도를 정밀하게 그려내는 '공식 사진 판독(Photo Finish)' 방식을 사용합니다.")
