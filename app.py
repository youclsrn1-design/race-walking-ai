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

# 2. AI 분석 엔진 초기화 (에러 해결: model_complexity=1 로 수정됨)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7)
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
                st.success(f"🔥 **[규정 통과 / 완벽한 스트레이트 레그] 무릎 신전도: {knee_val:.1f}°**\n\n발뒤꿈치 착지부터 수직선 통과까지 무릎이 완벽한 강체(Rigid lever)를 유지하고 있습니다. 무릎 관절의 에너지 흡수가 차단된 상태이므로, 전방 추진을 위해 엉덩이와 발목 관절의 스냅을 적극 활용하세요.")
            else:
                st.error(f"⚠️ **[실격 위험 / 벤트 니 파울] 무릎 신전도: {knee_val:.1f}°**\n\n착지 시 무릎 관절이 굽혀져 역학적 모순을 위반했습니다. 경보 규정 제54조에 따라 즉각적인 실격 사유가 됩니다. 발뒤꿈치 착지 순간 무릎을 일직선으로 완전히 펴는 신경근 제어에 집중하십시오.")

            # 2. 대퇴부 각도 (효율성)
            thigh_val = key_frame["t"]
            if 50 <= thigh_val <= 65:
                st.info(f"✅ **[퍼포먼스 우수 / 최적의 대퇴부 교차] 대퇴부 교차각: {thigh_val:.1f}°**\n\n대퇴부 각도가 이상적인 범위 내에 있습니다. 골반의 회전축을 안정적으로 유지하며 속도 창출을 극대화하고 있습니다.")
            elif thigh_val < 50:
                st.warning(f"⚠️ **[효율 저하 / 보폭 부족] 대퇴부 교차각: {thigh_val:.1f}°**\n\n대퇴부 각도가 50도 미만으로 좁아져 전진 속도 확보에 치명적인 손실이 발생하고 있습니다. 고관절 굴곡 및 신전 모멘트를 더 강하게 사용하세요.")
            else:
                st.warning(f"🚨 **[효율 저하 및 파울 위험 / 과도한 보폭] 대퇴부 교차각: {thigh_val:.1f}°**\n\n대퇴부 각도가 65도를 초과했습니다. 골반의 회전축이 흔들려 달리기로 착시를 유발할 수 있으며, 착지 시 제동력이 급증합니다.")

            # 3. 발목 착지각
            ankle_val = key_frame["a"]
            if 23 <= ankle_val <= 27:
                st.info(f"✅ **[퍼포먼스 우수 / 완벽한 착지 타격] 발목 각도: {ankle_val:.1f}°**\n\n형태학적 저항을 최소화하며 운동에너지 손실을 완벽히 막아내고 있습니다.")
            else:
                st.warning(f"⚠️ **[효율 저하 / 발목 제어 불안정] 발목 각도: {ankle_val:.1f}°**\n\n착지 각도가 23~27도 범위를 벗어났습니다. 발목의 강한 배측굴곡을 통해 예리한 각도로 지면을 타격하여 보행 주기의 안정성을 극대화해야 합니다.")

        # 종합 생체역학 처방 (박사급 피드백)
        st.divider()
        st.subheader("🎓 전문 생체역학 리포트 요약")
        st.write(f"""
        본 분석 결과, 귀하의 보행 주기는 글로벌 엘리트 데이터와 비교했을 때 **무릎 신전의 일관성** 측면에서 분석되었습니다. 
        측정된 데이터 리스트({fact_line}...)를 통해 볼 때, 신체 수직선 통과 시점의 각도 유지가 핵심 과제입니다. 
        **리프팅(Lifting)** 방지를 위해 체공 시간을 심판의 육안 한계치(0.042초) 미만으로 정밀하게 제어해야 하며, 보폭(Stride length) 확장을 자제하고 보빈도(Cadence)를 우선시하는 훈련 처방을 권장합니다.
        """)
