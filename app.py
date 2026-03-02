import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import math

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Foul Catcher", layout="wide")
st.title("🚨 국제 심판 전용: Rule 54 파울 적발 AI")
st.markdown("##### 💡 오직 'Bent Knee(170도 붕괴)'와 'Loss of Contact(양발 체공)' 파울만 추적하여 증거 프레임을 화면에 박제합니다.")
st.warning("🔒 업로드된 영상은 파울 프레임 추출 직후 서버에서 즉각 영구 삭제됩니다.")
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

# 💡 안전망이 추가된 170도 기준선 그리기
def draw_170_degree_line(img, hip, knee, ankle):
    try:
        length = math.hypot(ankle[0] - knee[0], ankle[1] - knee[1])
        angle_hip_knee = math.atan2(hip[1] - knee[1], hip[0] - knee[0])
        target_angle = angle_hip_knee + math.radians(170)
        
        target_x = int(knee[0] + length * math.cos(target_angle))
        target_y = int(knee[1] + length * math.sin(target_angle))
        
        cv2.line(img, (int(knee[0]), int(knee[1])), (target_x, target_y), (255, 0, 0), 4)
        cv2.putText(img, "170 Limit", (target_x, target_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    except Exception:
        pass # 계산 오류 시 다운되지 않고 조용히 패스

# 3. 영상 업로드 및 파울 수색
st.error("⚠️ **10초 이내의 훈련 영상**을 올려주세요. 파울이 발생한 찰나의 순간을 AI가 잡아냅니다.")
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
        with st.spinner("🕵️‍♂️ AI 심판이 파울을 정밀 판독 중입니다... (에러 방지 시스템 가동)"):
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
                        return [landmark.x * w, landmark.y * h]
                    
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
                    
                    hip_center_x = (l_h[0] + r_h[0]) / 2
                    facing_right = nose_x > hip_center_x 
                    leading_is_left = (l_a[0] > r_a[0]) if facing_right else (l_a[0] < r_a[0])

                    annotated = img.copy()
                    mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # 🚨 1. Bent Knee 판독
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1) and cd_bent_knee == 0:
                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        
                        front_angle = calculate_angle(front_hip, front_knee, front_ankle)
                        
                        if front_angle <= 170.0:
                            draw_170_degree_line(annotated, front_hip, front_knee, front_ankle)
                            cv2.putText(annotated, f"BENT KNEE: {front_angle:.1f} deg", (30, 80), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                            foul_bent_knee_frames.append((front_angle, annotated.copy()))
                            cd_bent_knee = 15

                    # 🚨 2. 체공 파울 판독
                    if trend > 0 and prev_trend < 0 and stride_dist < (w * 0.1) and cd_contact == 0:
                        floating_threshold = h * 0.03 
                        if (global_ground_y - current_lowest_y) > floating_threshold:
                            cv2.line(annotated, (0, int(global_ground_y)), (w, int(global_ground_y)), (255, 0, 0), 3)
                            cv2.putText(annotated, "GROUND LINE", (10, int(global_ground_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            cv2.putText(annotated, "LOSS OF CONTACT!", (30, 150), 
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
        st.header("🎯 Rule 54 심판 판독 결과 및 증거 자료")
        
        # 💡 [핵심 에러 해결] 화면 분할 로직 수정 (가로 한 줄에 몽땅 넣지 않고, 3개씩 예쁘게 줄바꿈)
        st.subheader(f"🔴 Bent Knee 파울 적발 횟수: 총 {len(foul_bent_knee_frames)}회")
        if len(foul_bent_knee_frames) > 0:
            st.error("⚠️ 앞다리 착지 순간 무릎이 170도 이하로 붕괴되었습니다. (빨간선이 170도 기준입니다)")
            
            # 3장씩 끊어서 출력하여 Streamlit 에러 방지
            for i in range(0, len(foul_bent_knee_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(foul_bent_knee_frames):
                        foul = foul_bent_knee_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 #{i+j+1} (무릎 각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 착지 시 170도 이상의 무릎 신전을 훌륭하게 유지했습니다.")
            
        st.write("---")
        
        st.subheader(f"🟡 Loss of Contact 파울 적발 횟수: 총 {len(foul_contact_frames)}회")
        if len(foul_contact_frames) > 0:
            st.warning("⚠️ 교차 순간 두 발이 모두 지면(빨간선)에서 떨어지는 체공 현상이 포착되었습니다.")
            
            for i in range(0, len(foul_contact_frames), 3):
                cols2 = st.columns(3)
                for j in range(3):
                    if i + j < len(foul_contact_frames):
                        img = foul_contact_frames[i + j]
                        with cols2[j]:
                            st.image(img, channels="RGB", caption=f"체공 파울 #{i+j+1}")
        else:
            st.success("✅ Loss of Contact 통과: 교차 구간에서도 지면 접촉을 잘 유지했습니다.")

st.write("---")
st.info("💡 **이 판독기는 국제 육상 연맹(World Athletics) Rule 54의 핵심인 '앞다리 신전'과 '교차 순간 체공'만을 수학적으로 필터링하여 증거 프레임을 추출합니다.**")
