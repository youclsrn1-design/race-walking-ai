import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Foul Catcher", layout="wide")
st.title("🚨 국제 심판 전용: Rule 54 절대 직선 검증 AI")
st.markdown("##### 💡 골반과 복숭아뼈를 잇는 '절대 직선'을 그어 무릎이 170도 이하로 무너지는 진짜 파울만 낚아챕니다.")
st.warning("🔒 업로드된 영상은 파울 프레임 추출 직후 서버에서 즉각 영구 삭제됩니다.")
st.write("---")

# 2. AI 분석 엔진 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

# 3. 영상 업로드
st.error("⚠️ **10초 이내의 훈련 영상**을 올려주세요. 실제 움직이는 앞다리만 정확하게 타겟팅합니다.")
video_file = st.file_uploader("경보 역학 분석 영상 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    foul_bent_knee_frames = []   
    foul_contact_frames = []     
    
    prev_stride_dist = 0
    prev_trend = 0
    prev_hip_center_x = 0 # 💡 이동 방향 추적용 변수 추가
    
    recent_lowest_ys = [] 
    
    cd_bent_knee = 0
    cd_contact = 0
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("🕵️‍♂️ AI가 실제 이동 방향을 추적하여 정확히 '앞다리'에만 기준선을 긋고 있습니다..."):
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
                    
                    l_h = get_pt(lm[mp_pose.PoseLandmark.LEFT_HIP])
                    r_h = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                    l_k = get_pt(lm[mp_pose.PoseLandmark.LEFT_KNEE])
                    r_k = get_pt(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                    l_a = get_pt(lm[mp_pose.PoseLandmark.LEFT_ANKLE])
                    r_a = get_pt(lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
                    
                    # 지면 추적
                    current_lowest_y = max(l_a[1], r_a[1])
                    recent_lowest_ys.append(current_lowest_y)
                    if len(recent_lowest_ys) > 10:
                        recent_lowest_ys.pop(0)
                    dynamic_ground_y = sum(recent_lowest_ys) / len(recent_lowest_ys)
                        
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    
                    # 💡 [핵심 알고리즘 수정] 코 위치가 아니라 골반의 '실제 픽셀 이동 궤적'으로 앞다리 판단
                    hip_center_x = (l_h[0] + r_h[0]) / 2
                    
                    # 초기 프레임 설정
                    if prev_hip_center_x == 0:
                        prev_hip_center_x = hip_center_x
                        
                    # 골반 중심이 오른쪽으로 이동했으면 우측 이동, 아니면 좌측 이동
                    moving_right = hip_center_x > prev_hip_center_x
                    
                    if moving_right:
                        leading_is_left = l_a[0] > r_a[0] # 오른쪽으로 걸을 땐 X좌표가 더 큰 쪽이 앞다리
                    else:
                        leading_is_left = l_a[0] < r_a[0] # 왼쪽으로 걸을 땐 X좌표가 더 작은 쪽이 앞다리

                    annotated = img.copy()

                    # =========================================================
                    # 🚨 1. 앞다리 타겟팅 완벽 적용: 골반-복숭아뼈 절대 직선 검증
                    # =========================================================
                    # 보폭이 넓어지는 착지 순간에만 포착
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1) and cd_bent_knee == 0:
                        # 확실한 앞다리(Front Leg) 매핑
                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        
                        front_angle = calculate_angle(front_hip, front_knee, front_ankle)
                        
                        # 앞다리 무릎이 170도 이하라면
                        if front_angle <= 170.0:
                            # 💡 진짜 앞다리에만 골반-복숭아뼈 빨간 직선 그리기
                            cv2.line(annotated, tuple(front_hip), tuple(front_ankle), (255, 0, 0), 4) # 두꺼운 빨간 줄
                            
                            # 무릎 꺾임 위치 노란 점
                            cv2.circle(annotated, tuple(front_knee), 10, (255, 255, 0), -1) 
                            
                            # 실제 꺾인 다리 뼈대 파란선
                            cv2.line(annotated, tuple(front_hip), tuple(front_knee), (0, 0, 255), 3)
                            cv2.line(annotated, tuple(front_knee), tuple(front_ankle), (0, 0, 255), 3)
                            
                            cv2.putText(annotated, f"FRONT LEG BENT: {front_angle:.1f} deg", (30, 80), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                            
                            foul_bent_knee_frames.append((front_angle, annotated.copy()))
                            cd_bent_knee = 15

                    # =========================================================
                    # 🚨 2. 진짜 점프(체공) 감지 로직
                    # =========================================================
                    if trend > 0 and prev_trend < 0 and stride_dist < (w * 0.1) and cd_contact == 0:
                        jump_threshold = h * 0.05 
                        if (dynamic_ground_y - current_lowest_y) > jump_threshold:
                            cv2.putText(annotated, "TRUE LOSS OF CONTACT!", (30, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                            foul_contact_frames.append(annotated.copy())
                            cd_contact = 15

                    prev_stride_dist = stride_dist
                    prev_trend = trend
                    prev_hip_center_x = hip_center_x # 골반 위치 지속 업데이트

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 판독 결과 출력
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("🎯 Rule 54 절대 직선 검증 리포트 (전방 다리 타겟팅 완료)")
        
        st.subheader(f"🔴 Bent Knee 파울 적발: 총 {len(foul_bent_knee_frames)}회")
        if len(foul_bent_knee_frames) > 0:
            st.error("⚠️ 실제 내딛는 앞다리의 골반-복숭아뼈 빨간선(가이드라인)에서 무릎이 벗어났습니다.")
            
            for i in range(0, len(foul_bent_knee_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(foul_bent_knee_frames):
                        foul = foul_bent_knee_frames[i + j]
                        with cols[j]:
                            st.image(foul[1], channels="RGB", caption=f"파울 #{i+j+1} (무릎: {foul[0]:.1f}°)")
        else:
            st.success("✅ Bent Knee 통과: 앞다리 무릎이 빨간선(170도 이상)에 훌륭하게 밀착해 있습니다.")
            
        st.write("---")
        
        st.subheader(f"🟡 Loss of Contact 파울 적발: 총 {len(foul_contact_frames)}회")
        if len(foul_contact_frames) > 0:
            st.warning("⚠️ 두 발이 명백하게 지면에서 솟구쳐 오르는 도약 현상이 감지되었습니다.")
            
            for i in range(0, len(foul_contact_frames), 3):
                cols2 = st.columns(3)
                for j in range(3):
                    if i + j < len(foul_contact_frames):
                        img = foul_contact_frames[i + j]
                        with cols2[j]:
                            st.image(img, channels="RGB", caption=f"체공 파울 #{i+j+1}")
        else:
            st.success("✅ Loss of Contact 통과: 한 발이 땅에 확실히 닿아있어 체공 오심을 완벽히 방어했습니다.")

st.write("---")
st.info("💡 **[전방 다리 타겟팅 알고리즘 가동 중]** 뒷다리 오심을 제거하고, 움직이는 '진짜 앞다리'에만 골반-복숭아뼈 절대 직선을 긋습니다.")
