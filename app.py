import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

# 1. 전문가용 인터페이스 설정
st.set_page_config(page_title="World Athletics AI Judge", layout="wide")
st.title("🔬 Pro Racewalking AI Judge & Biomechanics Lab")
st.error("⚠️ **[필독] 10초 이내의 짧은 영상을 업로드해 주세요.**")
st.markdown("##### 본 시스템은 World Athletics 규정을 준수하며, 측면(파울 판독)과 정면(리듬/효율 판독)을 모두 분석합니다.")

# 촬영 구도 선택 (측면 vs 정면)
view_mode = st.radio("🎥 영상 촬영 구도 선택", ["측면 (Side View) - 무릎 파울 및 착지각 판독", "정면 (Front View) - 골반 리듬 및 좌우 밸런스 판독"], horizontal=True)

# 2. AI 분석 엔진 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360-deg if deg > 180 else deg

def get_thigh_angle(h_l, k_l, h_r, k_r):
    v1, v2 = np.array(k_l) - np.array(h_l), np.array(k_r) - np.array(h_r)
    dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

def get_tilt_angle(p1, p2):
    # 두 점 사이의 수평 대비 기울기(리듬감) 계산
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    return np.degrees(np.arctan2(dy, dx))

# 3. 데이터 수집 및 분석 프로세스
video_file = st.file_uploader("경보 분석 영상 업로드", type=['mp4', 'mov'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # 변수 초기화
    knee_stats, hip_tilt_stats = [], []
    key_frame = None
    max_knee = 0
    max_hip_tilt = 0

    with st.spinner("AI가 프레임별 정밀 역학 판독을 수행 중입니다..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img)
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                
                # 좌표 추출
                l_h = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                r_h = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                l_k = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                r_k = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                l_a = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                l_f = [lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]

                annotated = img.copy()
                mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # [오류 해결 핵심] PIL Image로 변환하여 저장
                safe_img = Image.fromarray(annotated)

                if "측면" in view_mode:
                    # 측면 로직 (기존)
                    k_ang = calculate_angle(l_h, l_k, l_a)
                    a_ang = calculate_angle(l_k, l_a, l_f)
                    t_ang = get_thigh_angle(l_h, l_k, r_h, r_k)
                    knee_stats.append(k_ang)

                    if k_ang > max_knee:
                        max_knee = k_ang
                        key_frame = {"img": safe_img, "k": k_ang, "a": a_ang, "t": t_ang}
                
                else:
                    # 정면 로직 (골반 리듬 / 힙 드롭)
                    # 좌우 골반의 수평 기울기를 계산하여 리듬감 측정
                    tilt = abs(get_tilt_angle(l_h, r_h))
                    hip_tilt_stats.append(tilt)

                    if tilt > max_hip_tilt:
                        max_hip_tilt = tilt
                        key_frame = {"img": safe_img, "tilt": tilt}

    cap.release()
    os.unlink(tfile.name) # 영상 영구 삭제

    # 4. 박사급 전문 리포트 출력
    if key_frame:
        st.divider()
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("🎯 포착된 핵심 역학 프레임")
            # 에러가 나던 부분을 완벽히 해결한 코드
            st.image(key_frame["img"], use_container_width=True)
            
            if "측면" in view_mode:
                st.markdown("**무릎 신전 데이터 (최근 프레임):**")
                st.code(" / ".join([f"{ang:.1f}°" for ang in knee_stats[-8:]]))
            else:
                st.markdown("**골반 수직 이동(Hip Drop) 기울기 데이터:**")
                st.code(" / ".join([f"{tilt:.1f}°" for tilt in hip_tilt_stats[-8:]]))

        with col2:
            st.subheader("⚖️ 글로벌 엘리트 기준 기술 판독")
            
            if "측면" in view_mode:
                # 측면 판독 결과
                if key_frame["k"] >= 178:
                    st.success(f"🔥 **[통과] 무릎 신전도: {key_frame['k']:.1f}°**\n\n완벽한 강체(Rigid lever)를 유지하고 있습니다.")
                else:
                    st.error(f"⚠️ **[위험] 무릎 신전도: {key_frame['k']:.1f}° (Bent Knee)**\n\n실격 사유가 감지됩니다.")
                st.info(f"✅ **대퇴부 교차각: {key_frame['t']:.1f}°** (이상적 범위: 50~65°)")
                st.warning(f"⚠️ **발목 착지각: {key_frame['a']:.1f}°** (글로벌 기준: 23~27°)")

            else:
                # 정면 판독 결과 (리듬감)
                tilt_val = key_frame["tilt"]
                st.metric("최대 골반 기울기 (Max Pelvic Drop)", f"{tilt_val:.1f}°")
                
                if tilt_val < 5:
                    st.error("⚠️ **[리듬 결여 / 상하 진동 발생]**\n\n골반의 수직 이동(Hip Drop)이 거의 없습니다. 이는 충격을 골반이 흡수하지 못하고 어깨와 머리로 전달하여 상하 진동을 유발합니다. 고관절의 유연성을 활용하세요.")
                elif 5 <= tilt_val <= 12:
                    st.success("🔥 **[최적의 역학 리듬]**\n\n이상적인 골반 기울기(5~12도)를 보여줍니다. 체중 이동이 물 흐르듯 자연스러우며, 양발이 교차할 때 완벽한 '시계추 리듬(Pendulum rhythm)'이 형성되고 있습니다.")
                else:
                    st.warning("⚠️ **[과도한 골반 붕괴]**\n\n골반이 12도 이상 과도하게 떨어집니다. 코어 근육(중둔근)이 버티지 못하고 무너지는 현상으로, 에너지 누수가 발생합니다.")

        # 종합 생체역학 처방
        st.divider()
        st.subheader("🎓 전문 생체역학 처방 요약")
        if "측면" in view_mode:
            st.write(f"글로벌 엘리트 데이터와 대조한 결과, 평균 무릎 신전도({np.mean(knee_stats):.1f}°)는 안정적이나 착지각 제어가 필요합니다. 리프팅 방지를 위해 보폭 확장을 억제하세요.")
        else:
            st.write(f"정면 리듬 분석 결과, 귀하의 평균 골반 교차 기울기는 **{np.mean(hip_tilt_stats):.1f}°**입니다. 경보는 골반으로 걷는 스포츠입니다. 양 어깨의 수평은 유지하되, 골반만 위아래로 리드미컬하게 교차하는 '어깨-골반 분리(Shoulder-Pelvis Dissociation)' 훈련에 집중하십시오.")
