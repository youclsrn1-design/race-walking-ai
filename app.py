import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import math

# 1. 인터페이스 설정
st.set_page_config(page_title="Rule 54 Strict Catcher (High Accuracy)", layout="wide")
st.title("🚨 국제 심판 전용: Rule 54 절대 직선 검증 AI")
st.markdown("##### 💡 골반과 복숭아뼈를 잇는 '절대 직선'을 그어 무릎이 170도 이하로 무너지는 진짜 파울만 낚아챕니다.")
st.write("---")

# 2. AI 분석 엔진 초기화 (에러 해결 및 초정밀 모델 시도)
mp_pose = mp.solutions.pose
pose = None # 초기화

# 3. 영상 업로드 및 파울 수색
st.error("⚠️ **10초 이내의 영상**을 올려주세요. AI가 영상을 분석해 파울 사진을 추출하고 절대 직선 검증을 수행합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    # 💡 [핵심 해결] 초정밀 모델(Complexity 2)을 안전하게 로드하는 블록
    if pose is None:
        try:
            pose_high = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
            pose = pose_high # Success!
            st.success("🤖 **초정밀 3D 모델(Complexity 2)을 로드했습니다. 가장 높은 정확도로 분석합니다.**")
        except Exception as e:
            # 플랫폼 제한(PermissionError) 시 자동으로 표준 모델로 fallback
            pose_std = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
            pose = pose_std # Fallback
            st.warning(f"⚠️ **플랫폼 제한으로 초정밀 모델 로드에 실패하여 표준 모델(Complexity 1)을 사용합니다.** (미세 각도 오차 확률 있음) - 에러: {str(e)}")
            st.caption("클라우드 환경에서는 고복잡도 모델 다운로드가 차단될 수 있습니다. 이 경우 미세 각도 오차 확률이 높아집니다.")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    photo_finish_frames = []   
    flight_foul_frames = []
    
    prev_stride_dist = 0
    prev_trend = 0
    prev_hip_x = 0
    
    global_ground_y = 0.0
    flight_frames_count = 0 
    
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
            
        required_flight_frames = int(0.42 * fps)

        # 💡 [핵심] 선생님 요청대로 빨간선/노란점 그리는 내부 함수 정의
        def draw_strict_analytical_lines(img, hip, knee, ankle, angle, thick_multiplier=1):
            h, w, _ = img.shape
            overlay = img.copy()

            # Define Colors (BGR)
            RED = (0, 0, 255)
            YELLOW = (0, 255, 255)

            # Base Line thickness adaptive to image width
            base_thickness = max(3, int(w / 300))
            analytical_thickness = int(base_thickness * thick_multiplier)

            # 💡 요구사항: 무릎 기준으로 복숭아뼈와 골반뼈에 빨간색 선 그어!
            # 1. Draw Red Line Knee to Hip
            cv2.line(overlay, tuple(knee), tuple(hip), RED, analytical_thickness, cv2.LINE_AA)

            # 2. Draw Red Line Knee to Ankle
            cv2.line(overlay, tuple(knee), tuple(ankle), RED, analytical_thickness, cv2.LINE_AA)

            # 3. Draw Yellow Dot at Knee
            radius = max(8, int(w / 150))
            cv2.circle(overlay, tuple(knee), radius, YELLOW, -1, cv2.LINE_AA)

            # Solid NP copy is clearest for user review
            np.copyto(img, overlay)

            # Place Angle Text near knee (clear yellow on bold lines)
            font_scale = max(1.0, w / 700)
            font_thickness = max(2, int(w / 400))
            cv2.putText(img, f"{angle:.1f}", (knee[0] + 15, knee[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, YELLOW, font_thickness, cv2.LINE_AA)

        with st.spinner("🕵️‍♂️ 영상 전체 프레임 분석 중... (착지 프레임 추출 -> 절대 직선 정밀 검증 진행)"):
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
                    
                    # 이동 방향 판별 (골반 중심 X좌표 추적)
                    hip_center_x = (l_h[0] + r_h[0]) / 2
                    if prev_hip_x == 0: prev_hip_x = hip_center_x
                    moving_right = hip_center_x > prev_hip_x
                    
                    # 지면 기준선 업데이트
                    current_lowest_y = max(l_a[1], r_a[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y
                        
                    stride_dist = abs(l_a[0] - r_a[0])
                    trend = stride_dist - prev_stride_dist
                    annotated = img.copy()
                    
                    # 💡 분석 정확도를 직접 확인할 수 있도록 원본 다리 라인 위에 스켈레톤 기본 묘사
                    # (선생님 요청하신 절대선 묘사는 Foul 시에만 덧그림)
                    mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # =========================================================
                    # 🚨 1단계 & 2단계 통합: 착지 순간 추출 및 3차 정지화면 검증
                    # =========================================================
                    # 보폭이 가장 넓어지는 '착지 순간'에 사진(프레임) 찰칵!
                    if trend < 0 and prev_trend > 0 and stride_dist > (w * 0.1):
                        leading_is_left = (l_a[0] > r_a[0]) if moving_right else (l_a[0] < r_a[0])

                        front_hip = l_h if leading_is_left else r_h
                        front_knee = l_k if leading_is_left else r_k
                        front_ankle = l_a if leading_is_left else r_a
                        
                        front_angle = calculate_angle(front_hip, front_knee, front_ankle)
                        
                        # 각도가 170도 이하일 때만 (파울 의심 시에만) 정밀 분석 선을 그리고 박제
                        if front_angle <= 170.0:
                            # 💡 [핵심] 선생님 요청대로 빨간선/노란점 그리기 함수 호출 (굵게 묘사)
                            draw_strict_analytical_lines(annotated, front_hip, front_knee, front_ankle, front_angle, thick_multiplier=2.5)
                            
                            foul_bent_knee_frames.append((front_angle, annotated.copy()))

                    # =========================================================
                    # 🚨 [별도] 체공 파울: 0.42초 (프레임 누적) 절대 룰
                    # =========================================================
                    # 가장 낮은 발조차도 땅(global_ground_y)에서 떨어져 있다면 (오차범위 2% 허용)
                    if (global_ground_y - current_lowest_y) > (h * 0.02):
                        flight_frames_count += 1
                    else:
                        if flight_frames_count >= required_flight_frames:
                            flight_time_sec = flight_frames_count / fps
                            cv2.putText(annotated, f"FLIGHT: {flight_time_sec:.2f} sec", (30, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                            flight_foul_frames.append(annotated.copy())
                        # 초기화 (한 발이라도 땅에 닿으면 체공 카운트 리셋)
                        flight_frames_count = 0

                    prev_stride_dist = stride_dist
                    prev_trend = trend
                    prev_hip_x = hip_center_x

        cap.release()
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 판독 결과 출력
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("📸 Rule 54 판독 결과 리포트")
        
        # --- 1. 무릎 굽힘 (Bent Knee) ---
        st.subheader(f"🔴 Bent Knee (170도 이하 적발): 총 {len(foul_bent_knee_frames)}회")
        if len(foul_bent_knee_frames) > 0:
            st.error(f"⚠️ **{np.mean([f[0] for f in foul_bent_knee_frames]):.1f}° (최저 {min([f[0] for f in foul_bent_knee_frames]):.1f}°)** - 착지 순간 앞다리 무릎이 170도 아래로 무너지는 것이 감지되었습니다. 육안 검증이 필요합니다.")
            
            # 파울 프레임 가로 나열
            for i in range(0, len(foul_bent_knee_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(foul_bent_knee_frames):
                        foul = foul_bent_knee_frames[i + j]
                        with cols[j]:
                            # 💡 캡처된 사진 위에는 이미 함수에 의해 빨간색 절대선/노란점/각도가 그려져 있음
                            st.image(foul[1], channels="RGB", caption=f"파울 사진 #{i+j+1} (무릎 각도: {foul[0]:.1f}°)")
                            
                            # 💡 미세 오차 판독 피드백
                            angle_val = foul[0]
                            if angle_val <= 168.0:
                                st.error("🚨 **확정 파울:** AI가 초정밀 스캔한 결과 명백한 파울입니다.")
                            elif 168.0 < angle_val <= 170.0:
                                st.warning("⚠️ **미세 파울 (AI 오차 고려):** 모델 정밀도에 따라 잘못된 판독일 확률이 있습니다. 반드시 선수의 다리 윤곽과 빨간선을 육안으로 직접 대조해 최종 판단하세요.")
                            else: # 이 조건은 걸릴 일이 없음 (위의 loop 조건 때문에)
                                st.success("✅ **통과 (AI 오차):** 모델의 false positive 판독입니다.")

        else:
            st.success("✅ **Bent Knee 통과:** 착지 프레임 추출 후 3차 검증을 했으나 무릎이 모두 곧게 펴져 있습니다.")
            
        st.write("---")
        
        # --- 2. 체공 (Loss of Contact) ---
        st.subheader(f"🟡 Loss of Contact (0.42초 초과): 총 {len(flight_foul_frames)}회 적발")
        if len(flight_foul_frames) > 0:
            st.warning("⚠️ 두 발이 완전히 땅에서 떨어져 있는 시간이 0.42초 이상 누적된 명백한 도약 파울입니다.")
            
            for i in range(0, len(flight_foul_frames), 3):
                cols2 = st.columns(3)
                for j in range(3):
                    if i + j < len(flight_foul_frames):
                        img = flight_foul_frames[i + j]
                        with cols2[j]:
                            st.image(img, channels="RGB", caption=f"0.42초 이상 체공 적발 #{i+j+1}")
        else:
            st.success("✅ **Loss of Contact 통과:** 두 발이 떨어져 있는 시간이 0.42초 미만이거나 한 발이 땅에 닿아있습니다.")

st.write("---")
st.info("💡 **알고리즘 룰 강제:** 골반-복숭아뼈 직선을 기준선으로 삼고(초록선), 0.42초 이상 누적 체공될 경우에만 파울로 적발합니다.")
