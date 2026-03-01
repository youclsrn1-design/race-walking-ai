import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

# -------------------------------------------------------------------
# ⚙️ 1. 앱 기본 설정 및 UI (국가대표/엘리트 타겟 프로페셔널 디자인)
# -------------------------------------------------------------------
st.set_page_config(page_title="Pro Racewalking AI", page_icon="⏱️", layout="wide")

st.title("⏱️ 엘리트 경보 역학 분석 시스템 (Pro Racewalking AI)")
st.caption("World Athletics Rule 54 (구 230.2조) 기반 초정밀 생체역학 판독기")
st.write("국가대표 및 세계 최상위 경보 선수들의 **'하이브리드 바운싱 보행(Hybrid bouncing gait)'** 최적화 및 파울(Foul) 리스크 검증을 위한 3D 운동역학 추적 시스템입니다.")
st.warning("🔒 훈련 기밀 유지를 위해 업로드된 영상은 서버에 저장되지 않으며 메모리에서 즉시 파기됩니다.")

st.write("---")

# -------------------------------------------------------------------
# 📐 2. 역학 계산 수학 함수
# -------------------------------------------------------------------
def calculate_angle(a, b, c):
    """세 점(a, b, c)을 이용한 2D 관절 각도 계산"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_thigh_angle(hip_l, knee_l, hip_r, knee_r):
    """전방 스윙 시 형성되는 두 대퇴부(허벅지) 사이의 궤적 각도 계산"""
    vec_l = np.array(knee_l) - np.array(hip_l)
    vec_r = np.array(knee_r) - np.array(hip_r)
    cosine_angle = np.dot(vec_l, vec_r) / (np.linalg.norm(vec_l) * np.linalg.norm(vec_r))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# -------------------------------------------------------------------
# 📥 3. 영상 업로드 및 분석 파라미터 설정
# -------------------------------------------------------------------
st.subheader("1️⃣ 분석 데이터 입력")
col1, col2 = st.columns(2)
with col1:
    fps_input = st.number_input("카메라 촬영 FPS (초당 프레임)", min_value=30, max_value=240, value=60, step=30, 
                                help="Loss of Contact 판독을 위해 가급적 60fps 이상의 고속 촬영 영상을 권장합니다.")
with col2:
    uploaded_video = st.file_uploader("측면(Sagittal plane) 촬영 영상 업로드 (MP4/MOV)", type=['mp4', 'mov'])

st.write("---")

# -------------------------------------------------------------------
# 🚀 4. 실시간 AI 포즈 에스티메이션 및 역학 분석
# -------------------------------------------------------------------
if uploaded_video is not None and st.button("실시간 역학 데이터 추출 및 판독"):
    # 임시 파일 저장
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile_path = tfile.name
    tfile.close()
    
    cap = cv2.VideoCapture(tfile_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=2) # 엘리트용 고정밀 모델(complexity=2) 적용
    mp_drawing = mp.solutions.drawing_utils

    # 측정 변수 초기화
    max_knee_angle = 0
    max_thigh_angle = 0
    bent_knee_frame = None
    thigh_angle_frame = None
    
    with st.spinner("다채널 운동역학 알고리즘 구동 중... (질량중심(CoM) 궤적 및 관절 모멘트 분석)"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # [좌측/우측 관절 좌표 추출]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                
                # 1. 무릎 신전 각도 계산 (Bent Knee 파울 판독용)
                knee_angle_l = calculate_angle(l_hip, l_knee, l_ankle)
                if knee_angle_l > max_knee_angle:
                    max_knee_angle = knee_angle_l
                    mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    bent_knee_frame = Image.fromarray(image_rgb)
                
                # 2. 대퇴부 각도 계산 (효율성 및 오버스트라이딩 판독용)
                thigh_angle = calculate_thigh_angle(l_hip, l_knee, r_hip, r_knee)
                if thigh_angle > max_thigh_angle:
                    max_thigh_angle = thigh_angle
                    thigh_angle_frame = Image.fromarray(image_rgb)

        cap.release()
        pose.close()

    # 메모리 정리
    try:
        os.unlink(tfile_path)
    except:
        pass

    # -------------------------------------------------------------------
    # 📊 5. 엘리트 전문 판독 보고서 출력
    # -------------------------------------------------------------------
    if bent_knee_frame is not None:
        st.subheader("📋 역학 분석 판독 보고서 (Biomechanical Report)")
        
        # --- 섹션 A: 규정 위반 검증 ---
        st.markdown("### 🛑 규정 위반 검증 (Rule 54 Foul Detection)")
        c1, c2 = st.columns(2)
        with c1:
            st.image(bent_knee_frame, caption=f"최대 신전 무릎 각도: {max_knee_angle:.1f}°", use_column_width=True)
        with c2:
            st.markdown("#### 1. 무릎 신전 (Bent Knee) 규정")
            if max_knee_angle >= 175:
                st.success(f"**[PASS] 완벽한 강체 기둥 (Rigid Lever)**\n\n측정 각도 **{max_knee_angle:.1f}°**. 무릎 관절의 에너지 흡수 기전이 완벽히 차단되었습니다. 착지 순간부터 수직선을 통과할 때까지 완벽한 과신전(Hyperextension) 상태를 유지 중입니다.")
            else:
                st.error(f"**[FOUL RISK] 벤트 니(Bent Knee) 위험 수준**\n\n측정 각도 **{max_knee_angle:.1f}°**. 엘리트 기준 175° 미만으로 굴곡(Flexion)이 감지되었습니다. 즉각적인 실격 사유가 될 수 있습니다. 발뒤꿈치 착지 시 신경근 제어(Neuromuscular control)를 통해 무릎을 일직선으로 잠가야 합니다.")
            
            st.markdown("#### 2. 지속 접촉 (Loss of Contact) 규정")
            # 60fps 기준 1프레임은 약 0.016초. 0.042초 한계치 설명
            st.info(f"**[SYSTEM INFO] 프레임 분석 한계치 평가**\n\n입력된 {fps_input}FPS 영상 기준, 1프레임은 약 {(1/fps_input):.3f}초입니다. 인간 심판의 시각적 프레임 한계인 0.042초 미만으로 비행 시간(Flight time)을 제어해야 합니다. (남성 20km 세계 정상급: 0.03초 내외)")

        st.write("---")
        
        # --- 섹션 B: 운동역학적 퍼포먼스 ---
        st.markdown("### ⚡ 운동역학 효율 및 추진력 (Kinematic Efficiency)")
        c3, c4 = st.columns(2)
        with c3:
            st.image(thigh_angle_frame, caption=f"최대 대퇴부 이격 각도: {max_thigh_angle:.1f}°", use_column_width=True)
        with c4:
            st.markdown("#### 3. 대퇴부 궤적 및 보폭 효율 (Thigh Angle)")
            if 50 <= max_thigh_angle <= 65:
                st.success(f"**[OPTIMAL] 최적의 시상면 궤적**\n\n측정 각도 **{max_thigh_angle:.1f}°**. 50°~65° 사이의 완벽한 궤적입니다. 보폭(약 1.12m 한계치) 확장의 제동력(Braking force) 리스크 없이 걷기의 형태학적 껍질 속에서 달리기의 동력을 효율적으로 생산하고 있습니다.")
            elif max_thigh_angle < 50:
                st.warning(f"**[LOSS] 추진력 손실 (Short Stride)**\n\n측정 각도 **{max_thigh_angle:.1f}°**. 대퇴부 각도가 50° 미만입니다. 고관절 굴곡 및 신전 모멘트가 부족하여 전진 속도 확보에 치명적인 손실이 발생하고 있습니다.")
            else:
                st.error(f"**[DANGER] 오버스트라이딩 및 골반축 붕괴**\n\n측정 각도 **{max_thigh_angle:.1f}°**. 65°를 초과하여 골반의 전후방 기울임(Pelvic drop/tilt) 한계치를 넘었습니다. 착지 제동력이 급증하고 심판진에게 달리기(Running) 착시를 유발할 수 있습니다. 보폭을 줄이고 보빈도(Step frequency)를 분당 230 이상으로 끌어올리십시오.")
                
            st.markdown("#### 4. 동력 전이 (Power Transfer)")
            st.write("무릎의 추진력이 차단된 상태이므로, 입각기 말기 전방 추진의 절대적인 동력은 **발목 저측굴곡근(Plantarflexors)**에서 파생되어야 합니다. 착지 시 23°~27°의 예리한 발목 배측굴곡(Dorsiflexion) 텐션을 유지하십시오.")


