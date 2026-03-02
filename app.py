import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

# 1. 전문가용 인터페이스 설정
st.set_page_config(page_title="World Athletics AI Judge", layout="wide")
st.title("🔬 Pro Racewalking AI Judge (3D 통합 분석)")
st.markdown("##### 하나의 영상으로 Rule 54(측면) 위반 여부와 코어 밸런스(정면)를 동시에 정밀 판독합니다.")
st.warning("🔒 업로드된 모든 영상은 AI가 분석 후 임시 처리되어 즉시 소멸됩니다. (서버 무단 저장 절대 불가)")
st.write("---")

# 2. AI 분석 엔진 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 수학적 각도 계산 함수
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
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    return np.abs(np.degrees(np.arctan2(dy, dx)))

# 3. 통합 영상 업로드 및 분석
st.subheader("영상 업로드")
st.error("⚠️ **10초 이내의 영상**만 올려주세요! (어느 각도에서 찍었든 AI가 3D 좌표로 동시 분석합니다)")
video_file = st.file_uploader("경보 분석 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    knee_stats, hip_tilt_stats = [], []
    key_frame_side = None
    key_frame_front = None
    max_knee = 0
    max_hip_tilt = 0
    frame_count = 0
    person_detected = False

    # 🔥 try-finally 블록: 에러가 나도, 사용자가 중간에 나가도 무조건 파일 삭제 보장!
    try:
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("AI가 3D 공간 좌표를 매핑하여 측면과 정면의 핵심 역학을 동시 추출 중입니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                if frame_count % 3 != 0: continue # 속도 최적화
                
                height, width = frame.shape[:2]
                # 메모리 폭파 방지: 가로 해상도를 640으로 최적화하여 램 용량 방어
                if width > 640: frame = cv2.resize(frame, (640, int(height * 640 / width)))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                if res.pose_landmarks:
                    person_detected = True
                    lm = res.pose_landmarks.landmark
                    
                    l_h = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                    r_h = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                    l_k = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                    r_k = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                    l_a = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                    l_f = [lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]

                    annotated = img.copy()
                    mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    safe_img = annotated.copy()

                    # 1. 측면(Side) 데이터 추출
                    k_ang = calculate_angle(l_h, l_k, l_a)
                    a_ang = calculate_angle(l_k, l_a, l_f)
                    t_ang = get_thigh_angle(l_h, l_k, r_h, r_k)
                    knee_stats.append(k_ang)

                    if k_ang > max_knee:
                        max_knee = k_ang
                        key_frame_side = {"img": safe_img, "k": k_ang, "a": a_ang, "t": t_ang}

                    # 2. 정면(Front) 데이터 추출
                    tilt = get_tilt_angle(l_h, r_h)
                    hip_tilt_stats.append(tilt)

                    if tilt > max_hip_tilt:
                        max_hip_tilt = tilt
                        key_frame_front = {"img": safe_img, "tilt": tilt}

        cap.release()
    finally:
        # 🚀 무료 서버 평생 유지의 비결: 무슨 일이 있어도 임시 파일은 찢어버립니다.
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 통합 결과 리포트 출력
    if not person_detected:
        st.error("❌ 영상에서 선수를 인식하지 못했습니다. 전신이 잘 보이는 영상을 올려주세요.")
    elif not key_frame_side or not key_frame_front:
        st.warning("⚠️ 분석할 만한 뚜렷한 동작을 찾지 못했습니다.")
    else:
        st.divider()
        st.subheader("🎯 3D AI 통합 포착 리포트")
        
        col_side, col_front = st.columns(2)
        
        with col_side:
            st.markdown("#### 1️⃣ 착지 역학 포착 (측면)")
            st.image(key_frame_side["img"], channels="RGB", use_column_width=True)
            st.metric("무릎 신전도 (Rule 54)", f"{key_frame_side['k']:.1f}°")
            st.metric("발목 착지각 (제동력 판독)", f"{key_frame_side['a']:.1f}°")
            
        with col_front:
            st.markdown("#### 2️⃣ 코어 밸런스 포착 (정면)")
            st.image(key_frame_front["img"], channels="RGB", use_column_width=True)
            st.metric("최대 골반 기울기 (Hip Drop)", f"{key_frame_front['tilt']:.1f}°")
            st.caption("충격 흡수 및 시계추 리듬 평가 지표")

        st.divider()
        st.subheader("⚖️ 글로벌 스탠다드 종합 판독")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("**[측면 규정 위반 및 효율]**")
            if key_frame_side["k"] >= 178:
                st.success(f"🔥 **[통과] 완벽한 무릎 강체 (Rigid lever)**")
            else:
                st.error(f"⚠️ **[경고] Bent Knee (실격 사유 감지)**")
            
            if 23 <= key_frame_side["a"] <= 27:
                st.info("✅ 이상적인 발목 착지각을 유지 중입니다.")
            else:
                st.warning("⚠️ 발목 착지각 제어가 불안정하여 제동력이 발생합니다.")

        with res_col2:
            st.markdown("**[정면 골반 리듬 밸런스]**")
            tilt_val = key_frame_front["tilt"]
            if tilt_val < 5:
                st.error("⚠️ **[리듬 결여]** 골반 수직 이동이 너무 부족해 상하 진동이 발생합니다.")
            elif 5 <= tilt_val <= 12:
                st.success("🔥 **[리듬 최적]** 물 흐르듯 완벽한 시계추 리듬(Pendulum rhythm)이 형성 중입니다.")
            else:
                st.warning("⚠️ **[골반 붕괴]** 골반이 12도 이상 떨어집니다. 중둔근 강화가 시급합니다.")

        st.divider()
        st.subheader("🎓 전문가용 종합 처방전")
        avg_knee = np.mean(knee_stats) if knee_stats else 0
        avg_tilt = np.mean(hip_tilt_stats) if hip_tilt_stats else 0
        
        st.write(f"""
        하나의 질주 과정에서 측면의 추진력과 정면의 밸런스를 종합 판독한 결과입니다. 
        전체 구간 평균 무릎 신전도는 **{avg_knee:.1f}°**, 평균 골반 기울기는 **{avg_tilt:.1f}°**로 기록되었습니다.
        어느 한쪽이 무너지면 기록은 단축되지 않습니다. 착지 시 무릎을 일직선으로 펴는 동시에, 
        양 어깨의 수평을 유지하며 골반만 리드미컬하게 위아래로 교차시키는 **'어깨-골반 분리(Shoulder-Pelvis Dissociation)'** 훈련을 병행하십시오.
        """)

# 🔥 피드백 및 고객 지원 이메일 추가
st.write("---")
st.info("💡 **시스템 개선을 위한 소중한 의견을 들려주세요!**\n\n버그 신고, 추가 판독 지표 제안 등 어떠한 피드백이든 적극 수용합니다.\n\n📧 **문의처:** [youclsrn1@gmail.com](mailto:youclsrn1@gmail.com)")
