import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 1. 국제 심판 모드 인터페이스 설정
st.set_page_config(page_title="Rule 54 Foul Catcher", layout="wide")
st.title("🚨 국제 심판 전용: Rule 54 파울 적발 AI")
st.markdown("##### 💡 오직 'Bent Knee(무릎 굽힘)'와 'Loss of Contact(양발 체공)' 파울만 추적하여 증거 프레임을 화면에 박제합니다.")
st.warning("🔒 업로드된 영상은 파울 프레임 추출 직후 서버에서 즉각 영구 삭제됩니다.")
st.write("---")

# 2. AI 분석 엔진 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

# 3. 영상 업로드 및 파울 수색
st.error("⚠️ **10초 이내의 영상**을 올려주세요. 파울이 발생한 찰나의 순간을 AI가 잡아냅니다.")
video_file = st.file_uploader("경보 역학 분석 영상 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    # 파울 증거 사진 저장소
    foul_bent_knee_frames = []   # (무릎 각도, 이미지)
    foul_contact_frames = []     # (이미지)
    
    # 지면 인식 및 보폭 추적 변수
    prev_stride_dist = 0
    prev_trend = 0
    ground_y = 0.0 # 지면 Y좌표 (화면 아래쪽일수록 값이 큼)
    
    # 연사 방지 쿨다운 (파울 한 번 잡으면 15프레임 동안은 중복 캡처 방지)
    cd_bent_knee = 0
    cd_contact = 0
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("🕵️‍♂️ AI 국제 심판이 영상을 프레임 단위로 쪼개어 파울을 적발하고 있습니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                h, w = frame.shape[:2]
                if w > 640: frame = cv2.resize(frame, (640, int(h * 640 / w)))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                if cd_bent_knee > 0: cd_bent_knee -= 1
                if cd_contact > 0: cd_contact -= 1
                
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
                    
                    # 지면 라인(가장 낮은 발목) 지속 갱신
                    current_lowest_y = max(l_a[1], r_a[1])
                    if current_lowest_y > ground_y:
                        ground_y = current_lowest_y
                        
                    # 보폭 변화 추적 (착지 순간과 교차 순간을 알기 위함)
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    
                    # 진행 방향에 따른 전방 다리(앞다리) 구별
                    hip_center_x = (l_h[0] + r_h[0]) / 2
                    facing_right = nose_x > hip_center_x 
                    leading_is_left = (l_a[0] > r_a[0]) if facing_right else (l_a[0] < r_a[0])

                    annotated = img.copy()
                    mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # =========================================================
                    # 🚨 1. 무릎 굽힘 (Bent Knee) 파울 적발 로직
                    # 보폭이 가장 넓어지는 '착지 순간(Contact)'에 앞다리 무릎이 170도 이하인가?
                    # =========================================================
                    if trend < 0 and prev_trend > 0 and stride_dist > 0.05 and cd_bent_knee == 0:
                        front_angle = calculate_angle(l_h, l_k, l_a) if leading_is_left else calculate_angle(r_h, r_k, r_a)
                        
                        # 170도 이하면 심판의 눈에 파울로 간주 -> 빨간 글씨 박제
                        if front_angle <= 170.0:
                            cv2.putText(annotated, f"BENT KNEE: {front_angle:.1f} deg", (20, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                            foul_bent_knee_frames.append((front_angle, annotated.copy()))
                            cd_bent_knee = 15 # 중복 캡처 방지 쿨다운

                    # =========================================================
                    # 🚨 2. 양발 체공 (Loss of Contact) 파울 적발 로직
                    # 보폭이 가장 좁아지는 '교차 순간(Mid-Stance)'에 양 발이 모두 허공에 떠 있는가?
                    # =========================================================
                    if trend > 0 and prev_trend < 0 and stride_dist < 0.08 and cd_contact == 0:
                        # 현재 프레임의 가장 낮은 발조차도 지면 라인(ground_y)보다 2.5% 이상 위로 떠 있다면 체공으로 간주
                        if (ground_y - current_lowest_y) > 0.025: 
                            cv2.putText(annotated, "LOSS OF CONTACT!", (20, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
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
        st.header("🎯 국제 심판 Rule 54 파울 적발 리포트")
        
        # --- 1. 무릎 굽힘 (Bent Knee) 박제 영역 ---
        st.subheader(f"🔴 Bent Knee 파울 적발 횟수: 총 {len(foul_bent_knee_frames)}회")
        if len(foul_bent_knee_frames) > 0:
            st.error("⚠️ 앞다리가 착지하는 순간 무릎이 170도 이하로 무너지는 파울이 적발되었습니다. 아래는 해당 프레임의 증거입니다.")
            
            # 파울 프레임 가로 나열
            cols = st.columns(len(foul_bent_knee_frames))
            for idx, foul in enumerate(foul_bent_knee_frames):
                with cols[idx]:
                    st.image(foul[1], channels="RGB", caption=f"파울 #{idx+1} (무릎 각도: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 착지 시 앞다리 무릎이 기준치(170도 이상)를 잘 유지했습니다.")
            
        st.write("---")
        
        # --- 2. 양발 체공 (Loss of Contact) 박제 영역 ---
        st.subheader(f"🟡 Loss of Contact 파울 적발 횟수: 총 {len(foul_contact_frames)}회")
        if len(foul_contact_frames) > 0:
            st.warning("⚠️ 두 다리가 교차하는 찰나의 순간, 지면에 닿은 발 없이 두 발이 모두 공중에 떠오른 파울이 적발되었습니다.")
            
            # 체공 프레임 가로 나열
            cols2 = st.columns(len(foul_contact_frames))
            for idx, img in enumerate(foul_contact_frames):
                with cols2[idx]:
                    st.image(img, channels="RGB", caption=f"체공 파울 #{idx+1}")
        else:
            st.success("✅ Loss of Contact 통과: 교차 구간에서도 항상 한 발이 지면에 접촉해 있습니다.")

st.write("---")
st.info("💡 본 시스템은 심판의 육안(To the human eye) 판단 기준에 의거, 착지 구간(Bent Knee)과 교차 구간(Loss of Contact)을 집중 감시하여 룰 위반 프레임을 자동 추출합니다.\n\n📧 **문의처:** [youclsrn1@gmail.com](mailto:youclsrn1@gmail.com)")
