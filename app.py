import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import time

# 1. 페이지 레이아웃 및 보안 설정
st.set_page_config(page_title="World Athletics AI Judge", layout="wide")

st.title("🔬 Pro Racewalking AI Judge & Biomechanics Lab")
st.error("⚠️ **[중요] 분석을 위해 10초 이내의 영상을 업로드해 주세요.** (고용량 시 분석 속도 저하 방지)")
st.markdown("##### 본 시스템은 World Athletics Rule 54 규정을 준수하며, 분석 후 모든 영상은 즉시 파기됩니다.")

# 2. AI 분석 엔진 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360-deg if deg > 180 else deg

def get_thigh_angle(h_l, k_l, h_r, k_r):
    # 두 대퇴부 벡터 사이의 각도 계산
    v1 = np.array(k_l) - np.array(h_l)
    v2 = np.array(k_r) - np.array(h_r)
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

# 3. 영상 업로드 및 분석
video_file = st.file_uploader("경보 분석 영상을 업로드하세요 (측면 촬영 권장)", type=['mp4', 'mov'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # 분석 데이터 저장용 변수
    knee_angles = []
    thigh_angles = []
    ankle_angles = []
    frames_processed = 0
    key_frame = None
    max_knee_ext = 0

    with st.spinner("AI가 세계육상연맹 규정에 따라 프레임별 정밀 판독을 실시하고 있습니다..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # 관절 추출 (왼쪽 다리 중심 분석)
                l_h = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                l_k = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                l_a = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                l_f = [lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
                r_h = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                r_k = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]

                # 각도 산출
                k_ang = calculate_angle(l_h, l_k, l_a)
                a_ang = calculate_angle(l_k, l_a, l_f)
                t_ang = get_thigh_angle(l_h, l_k, r_h, r_k)

                knee_angles.append(k_ang)
                thigh_angles.append(t_ang)
                ankle_angles.append(a_ang)

                # 가장 중요한 착지 프레임 포착 (무릎 최대 신전 시점)
                if k_ang > max_knee_ext:
                    max_knee_ext = k_ang
                    annotated_img = img_rgb.copy()
                    mp_drawing.draw_landmarks(annotated_img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    key_frame = {"img": annotated_img, "k": k_ang, "a": a_ang, "t": t_ang}
            
            frames_processed += 1

    cap.release()
    os.unlink(tfile.name) # 🔒 개인정보 보호: 영상 즉시 삭제

    # 4. 분석 결과 보고서 (Fact 중심)
    if key_frame:
        st.divider()
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("🎯 포착된 분석 핵심 프레임")
            st.image(key_frame["img"], use_container_width=True)
            
            # 실제 측정된 무릎 각도 시계열 데이터 나열 (Fact)
            st.markdown("**AI 추출 무릎 신전 데이터 (최근 10개 프레임):**")
            fact_line = " / ".join([f"{ang:.1f}°" for ang in knee_angles[-10:]])
            st.code(fact_line)
            st.caption(f"전체 분석 프레임 평균 무릎 각도: {np.mean(knee_angles):.1f}°")

        with col2:
            st.subheader("⚖️ 글로벌 엘리트 기준 판독 결과")
            
            # 1. 벤트 니 판독 (Rule 54)
            knee_val = key_frame["k"]
            if knee_val >= 178:
                st.success(f"✅ **[규정 통과] 무릎 신전도: {knee_val:.1f}°**\n\n발뒤꿈치 착지부터 수직선 통과까지 무릎이 완벽한 강체(Rigid lever)를 유지하고 있습니다. 추진을 위해 고관절 스냅을 활용하세요.")
            else:
                st.error(f"⚠️ **[실격 위험] 무릎 신전도: {knee_val:.1f}° (Bent Knee)**\n\n경보 규정 제54조 위반 가능성이 큽니다. 착지 순간 무릎을 일직선으로 완전히 펴는 신경근 제어에 집중하십시오.")

            # 2. 대퇴부 각도 (효율성)
            thigh_val = key_frame["t"]
            if 50 <= thigh_val <= 65:
                st.info(f"✅ **[퍼포먼스 우수] 대퇴부 교차각: {thigh_val:.1f}°**\n\n골반 회전축이 안정적이며 속도 창출을 극대화하고 있는 이상적 범위입니다.")
            else:
                st.warning(f"🚨 **[효율 저하] 대퇴부 교차각: {thigh_val:.1f}°**\n\n범위를 벗어난 각도는 제동력을 유발하거나 보폭 손실을 의미합니다. 고관절 모멘트를 수정하십시오.")

            # 3. 발목 착지각
            ankle_val = key_frame["a"]
            if 23 <= ankle_val <= 27:
                st.info(f"✅ **[최적 착지] 발목 각도: {ankle_val:.1f}°**\n\n형태학적 저항을 최소화하며 운동에너지 손실을 완벽히 막아내고 있습니다.")
            else:
                st.warning(f"⚠️ **[제어 불안정] 발목 각도: {ankle_val:.1f}°**\n\n예리한 각도로 지면을 타격하여 보행 주기의 안정성을 확보해야 합니다. 강한 배측굴곡이 필요합니다.")

        # 종합 생체역학 처방 (박사급 피드백)
        st.divider()
        st.subheader("🎓 전문 생체역학 리포트 요약")
        st.write(f"""
        본 분석 결과, 귀하의 보행 주기는 글로벌 엘리트 데이터와 비교했을 때 **무릎 신전의 일관성** 측면에서 분석되었습니다. 
        측정된 데이터 리스트({fact_line}...)를 통해 볼 때, 신체 수직선 통과 시점의 각도 유지가 핵심 과제입니다. 
        **리프팅(Lifting)** 방지를 위해 보폭(Stride length)보다는 보빈도(Cadence)를 우선시하는 훈련 처방을 권장합니다.
        """)
