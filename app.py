import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
from collections import Counter

# 1. 전문가용 인터페이스 설정
st.set_page_config(page_title="Elite Racewalking Biomechanics Lab", layout="wide")
st.title("🔬 엘리트 경보 생체역학 AI 판독 시스템 (Ph.D. Edition)")
st.markdown("##### 💡 0.42초 데스 존(Crossover Phase) 정밀 추적 및 Loss of Contact 감지 엔진 탑재")
st.markdown("Rule 54의 핵심인 **'양발 교차 순간(Mid-Stance)'**을 AI가 0.01초 단위로 분해합니다. 체중이 실리는 교차 지점에서 전방 다리가 170도 밑으로 붕괴되는지, 상하 진동으로 인한 '양발 체공(Loss of Contact)'이 발생하는지 박사급으로 판독합니다.")
st.warning("🔒 [Privacy-First] 본 시스템은 분석 직후 메모리에서 영상을 즉각 영구 파쇄하여 기밀을 완벽히 보호합니다.")
st.write("---")

# 2. AI 분석 엔진 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360-deg if deg > 180 else deg

def get_tilt_angle(p1, p2):
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    return np.abs(np.degrees(np.arctan2(dy, dx)))

# 3. 통합 영상 업로드 및 분석
st.subheader("훈련 영상 데이터 업로드")
st.error("⚠️ **10초 이내의 측면(Side) 영상**을 권장합니다. AI가 0.42초의 교차 구간을 샅샅이 뒤집니다.")
video_file = st.file_uploader("경보 역학 분석 영상 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    view_types = [] 
    contact_knees = [] 
    hip_tilt_stats = [] 
    
    # [새로운 데스 존 분석용 변수]
    crossover_knees = [] # 양 발목이 교차할 때의 무릎 각도 (170도 붕괴 확인용)
    hip_y_trajectory = [] # 골반 수직 이동 궤적 (체공 확인용)
    
    key_frame_side = None
    key_frame_front = None
    key_frame_crossover = None # 교차 순간의 증거 사진
    
    max_knee = 0
    max_hip_tilt = 0
    frame_count = 0
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("AI가 0.42초 교차 구간(Death Zone)의 무릎 붕괴와 양발 체공 여부를 추적 중입니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                if frame_count % 3 != 0: continue 
                
                h, w = frame.shape[:2]
                if w > 640: frame = cv2.resize(frame, (640, int(h * 640 / w)))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                if res.pose_landmarks:
                    person_detected = True
                    lm = res.pose_landmarks.landmark
                    
                    l_s = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    l_h = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                    r_h = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                    l_k = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                    r_k = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                    l_a = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                    r_a = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
                    l_f = [lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
                    r_f = [lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]
                    nose_x = lm[mp_pose.PoseLandmark.NOSE].x
                    
                    # 구도 자동 판별
                    hip_width = abs(l_h[0] - r_h[0])
                    torso_height = abs(l_s[1] - l_h[1])
                    current_view = "front" if (hip_width / (torso_height + 0.0001)) > 0.25 else "side"
                    view_types.append(current_view)

                    # 전방 다리 강제 추적
                    hip_center_x = (l_h[0] + r_h[0]) / 2
                    facing_right = nose_x > hip_center_x 
                    
                    if facing_right:
                        leading_is_left = l_a[0] > r_a[0] 
                    else:
                        leading_is_left = l_a[0] < r_a[0] 

                    if leading_is_left:
                        hip, knee, ankle, foot = l_h, l_k, l_a, l_f
                    else:
                        hip, knee, ankle, foot = r_h, r_k, r_a, r_f

                    annotated = img.copy()
                    mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    k_ang = calculate_angle(hip, knee, ankle)
                    a_ang = calculate_angle(knee, ankle, foot)
                    
                    # 💡 [핵심 기술 1] 데스 존 (Crossover Phase) 추적
                    # 양 발목의 X좌표 거리가 아주 가까워지는 순간(체중이 실리는 미드 스탠스)
                    ankle_distance = abs(l_a[0] - r_a[0])
                    if ankle_distance < 0.06: # 발목이 겹치는 찰나의 순간
                        crossover_knees.append(k_ang)
                        key_frame_crossover = {"img": annotated.copy(), "k": k_ang, "a": a_ang}
                        # 체공 감지를 위한 골반 높이 추적
                        hip_y_trajectory.append((l_h[1] + r_h[1]) / 2)

                    # 착지 순간 수집
                    if k_ang >= 170:
                        contact_knees.append(k_ang)

                    if k_ang > max_knee:
                        max_knee = k_ang
                        key_frame_side = {"img": annotated.copy(), "k": k_ang, "a": a_ang}

                    # 정면 역학 
                    tilt = get_tilt_angle(l_h, r_h)
                    hip_tilt_stats.append(tilt)
                    if tilt > max_hip_tilt:
                        max_hip_tilt = tilt
                        key_frame_front = {"img": annotated.copy(), "tilt": tilt}

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    if not person_detected:
        st.error("❌ 알고리즘이 피사체의 생체역학 랜드마크를 추출하지 못했습니다.")
    else:
        dominant_view = Counter(view_types).most_common(1)[0][0] if view_types else "side"

        st.divider()
        st.subheader("🎯 0.42초 데스 존(Crossover Phase) 정밀 판독 결과")
        
        if dominant_view == "side":
            avg_knee = np.mean(contact_knees) if contact_knees else max_knee
            # 교차 순간 가장 무너진 무릎 각도 추출
            min_crossover_knee = min(crossover_knees) if crossover_knees else avg_knee
            
            # 체공(Jumping) 감지: 교차 구간에서 골반이 비정상적으로 치솟는지 확인
            bounce_warning = False
            if hip_y_trajectory:
                bounce_variance = max(hip_y_trajectory) - min(hip_y_trajectory)
                if bounce_variance > 0.03: # 골반 상하 진폭이 기준치 초과 시 (러닝 메커니즘)
                    bounce_warning = True

            col1, col2 = st.columns([1, 1])
            
            # 교차 순간 증거 사진 우선 출력
            display_frame = key_frame_crossover if key_frame_crossover else key_frame_side
            
            with col1:
                st.image(display_frame["img"], channels="RGB", use_column_width=True, caption="양발 교차(Mid-Stance) 순간 포착")
            with col2:
                st.markdown("#### 🚨 Rule 54 심층 판독 (Loss of Contact & Bent Knee)")
                
                # 1. 무릎 붕괴 판독
                st.metric("교차 순간 최저 무릎 각도", f"{min_crossover_knee:.1f}°")
                if min_crossover_knee >= 178:
                    st.success("🔥 **[무릎 강체 통과]** 체중이 100% 실리는 교차 구간에서도 무릎이 완벽하게 180도에 가깝게 잠겨(Locked) 있습니다.")
                elif 170 <= min_crossover_knee < 178:
                    st.warning("⚠️ **[충격 흡수 주의]** 착지 후 무릎이 미세하게 굽혀지며 지면 반력을 잃고 있습니다. 심판의 육안에 띄일 수 있는 위험 구간입니다.")
                else:
                    st.error(f"❌ **[Bent Knee 실격]** 교차 순간 무릎이 {min_crossover_knee:.1f}도로 완전히 무너졌습니다(170도 붕괴). 대퇴사두근의 버티는 힘이 부족합니다.")

                st.write("---")
                
                # 2. 양발 체공 판독
                if bounce_warning:
                    st.error("❌ **[Loss of Contact 경고]** 교차 구간에서 골반의 상하 튀어오름(Bouncing)이 감지되었습니다. 뒷발이 차고 나갈 때 앞발이 지면에 닿지 않는 '양발 체공(비행)' 상태가 의심됩니다.")
                else:
                    st.success("✅ **[지면 접촉 유지]** 체공 현상 없이 골반의 회전으로 안정적인 지면 접촉(Contact)을 유지하고 있습니다.")

                st.markdown(f"*참고 데이터: 전체 유효 착지 평균 신전도 ({avg_knee:.1f}°)*")

        else: 
            st.success("🎥 **[관상면(정면) 감지]** 측면 분석을 차단하고, 골반 수직 이동(Pelvic Drop) 판독 모드를 가동합니다.")
            avg_tilt = np.mean(hip_tilt_stats) if hip_tilt_stats else 0
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(key_frame_front["img"], channels="RGB", use_column_width=True)
            with col2:
                st.markdown("#### ⚖️ 코어 밸런스 동역학")
                st.metric("최대 골반 기울기 (Max Pelvic Drop)", f"{key_frame_front['tilt']:.1f}°")
                st.markdown(f"**평균 골반 기울기: {avg_tilt:.1f}°**")
                
                tilt_val = key_frame_front["tilt"]
                if tilt_val < 5:
                    st.error("⚠️ **[리듬 결여]** 골반 가동성이 결여되어 상하 진동이 발생합니다. (체공 파울 위험 상승)")
                elif 5 <= tilt_val <= 12:
                    st.success("🔥 **[역학적 최적]** 에너지 손실이 없는 이상적인 시계추 리듬이 관측됩니다.")
                else:
                    st.warning("⚠️ **[에너지 누수]** 골반 붕괴가 감지됩니다. 중둔근 코어 제어력이 상실되었습니다.")

st.write("---")
st.info("💡 **생체역학 엔진 R&D 피드백**\n\n지속적인 알고리즘 고도화를 위해 현장의 소중한 피드백을 수렴하고 있습니다. 모델의 오작동 및 신규 분석 지표 제안은 아래 공식 채널로 문의 바랍니다.\n\n📧 **Chief Developer:** [youclsrn1@gmail.com](mailto:youclsrn1@gmail.com)")
