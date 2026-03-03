import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Rule 54 Freeze-Frame VAR", layout="wide")
st.title("🎬 Rule 54 방송용 VAR (3초 정지화면 탑재)")
st.markdown("##### 💡 1. 무릎 굽힘(Bent Knee): 파울 발생 순간 화면이 '3초간 일시 정지'하며, 아주 굵은 빨간선으로 시각을 고정시킵니다.")
st.markdown("##### 💡 2. 플라잉(Loss of Contact): 두 발이 떠 있는 순간은 보조선이 없는 '원본 사진'으로 박제합니다.")
st.write("---")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

st.error("⚠️ **10초 이내의 경보 영상**을 올려주세요. 정밀 판독 및 3초 정지 하이라이트 영상을 생성합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    out_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
    
    flight_foul_frames = [] 
    global_ground_y = 0.0
    flight_frames_count = 0 
    
    worst_bent_angle = 180.0 
    person_detected = False
    
    # 💡 [핵심] 3초 멈춤 쿨다운 변수 (한 걸음에 여러 번 멈추는 것 방지)
    freeze_cooldown = 0

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        scale_ratio = 800 / orig_w if orig_w > 800 else 1.0
        out_w = int(orig_w * scale_ratio)
        out_h = int(orig_h * scale_ratio)
        
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, out_h))
            
        required_flight_frames = int(0.08 * fps)
        if required_flight_frames < 2: required_flight_frames = 2

        with st.spinner("🕵️‍♂️ 파울 순간 화면을 3초간 정지시키고 굵은 빨간선을 칠하는 중입니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if scale_ratio != 1.0:
                    frame = cv2.resize(frame, (out_w, out_h))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                clean_frame = img.copy() 
                annotated = img.copy()   
                
                # 이번 프레임을 기본 4배속으로 기록할지 여부
                draw_normal_slowmo = True 
                
                if res.pose_landmarks:
                    person_detected = True
                    lm = res.pose_landmarks.landmark
                    
                    def get_pt(landmark):
                        return [int(landmark.x * out_w), int(landmark.y * out_h)]
                    
                    l_h = get_pt(lm[mp_pose.PoseLandmark.LEFT_HIP])
                    r_h = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                    l_k = get_pt(lm[mp_pose.PoseLandmark.LEFT_KNEE])
                    r_k = get_pt(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                    l_heel = get_pt(lm[mp_pose.PoseLandmark.LEFT_HEEL])
                    r_heel = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HEEL])
                    nose_x = lm[mp_pose.PoseLandmark.NOSE].x * out_w
                    
                    waist_center = [int((l_h[0] + r_h[0]) / 2), int((l_h[1] + r_h[1]) / 2)]
                    moving_right = nose_x > waist_center[0]
                    
                    front_is_left = (l_heel[0] > r_heel[0]) if moving_right else (l_heel[0] < r_heel[0])
                    
                    current_lowest_y = max(l_heel[1], r_heel[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y

                    # =========================================================
                    # 🚨 [1단계] 종골 착지 시 각도 변화 및 3초 정지(Freeze) 기능
                    # =========================================================
                    f_heel = l_heel if front_is_left else r_heel
                    f_knee = l_k if front_is_left else r_k
                    
                    is_in_front = (f_heel[0] > waist_center[0]) if moving_right else (f_heel[0] < waist_center[0])
                    is_grounded = abs(global_ground_y - f_heel[1]) < (out_h * 0.04)

                    if is_in_front and is_grounded:
                        current_angle = calculate_angle(waist_center, f_knee, f_heel)
                        
                        if current_angle < worst_bent_angle:
                            worst_bent_angle = current_angle
                            
                        if current_angle < 170.0:
                            if freeze_cooldown == 0:
                                # 💡 [하이라이트 연출] 170도 붕괴 첫 순간 -> 3초 멈춤 & 아주 굵은 빨간선!
                                line_color = (0, 0, 255)
                                line_thick_bold = max(10, int(out_w / 60)) # 기본 선보다 3배 굵게
                                
                                cv2.line(annotated, tuple(waist_center), tuple(f_knee), line_color, line_thick_bold, cv2.LINE_AA)
                                cv2.line(annotated, tuple(f_knee), tuple(f_heel), line_color, line_thick_bold, cv2.LINE_AA)
                                
                                # 타겟 포인트도 크게
                                cv2.circle(annotated, tuple(waist_center), 12, (255, 0, 255), -1) 
                                cv2.circle(annotated, tuple(f_knee), 12, (0, 255, 255), -1)      
                                cv2.circle(annotated, tuple(f_heel), 12, (0, 255, 0), -1)        
                                
                                # 경고 텍스트 추가
                                cv2.putText(annotated, f"FOUL: {current_angle:.1f} deg", (f_knee[0] + 30, f_knee[1]), 
                                            cv2.FONT_HERSHEY_SIMPLEX, max(1.2, out_w/600), line_color, 4)
                                cv2.putText(annotated, "VAR: 170 DEGREE FAILED!", (30, 60), 
                                            cv2.FONT_HERSHEY_SIMPLEX, max(1.0, out_w/700), line_color, 4)

                                bgr_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                                
                                # 💡 비디오에 이 프레임을 fps * 3 번 복사해서 넣음 (= 3초 동안 화면 정지)
                                for _ in range(int(fps * 3)): 
                                    out_video.write(bgr_frame)
                                    
                                freeze_cooldown = int(fps * 1.5) # 1.5초 동안은 다시 멈추지 않음 (다음 걸음으로 넘어갈 시간 부여)
                                draw_normal_slowmo = False # 3초 멈춤을 기록했으므로 아래 기본 기록은 건너뜀
                                
                            else:
                                # 쿨다운 중일 때는 (이미 한 번 멈췄으므로) 일반 굵기 빨간선으로 재생
                                line_color = (0, 0, 255)
                                line_thick = max(4, int(out_w / 180))
                                cv2.line(annotated, tuple(waist_center), tuple(f_knee), line_color, line_thick, cv2.LINE_AA)
                                cv2.line(annotated, tuple(f_knee), tuple(f_heel), line_color, line_thick, cv2.LINE_AA)
                                cv2.circle(annotated, tuple(waist_center), 8, (255, 0, 255), -1) 
                                cv2.circle(annotated, tuple(f_knee), 8, (0, 255, 255), -1)      
                                cv2.circle(annotated, tuple(f_heel), 8, (0, 255, 0), -1)        
                                cv2.putText(annotated, f"{current_angle:.1f} deg", (f_knee[0] + 20, f_knee[1]), 
                                            cv2.FONT_HERSHEY_SIMPLEX, max(0.8, out_w/900), line_color, 3)
                        else:
                            # 170도 이상 유지 시 (안전 구간) 초록선
                            line_color = (0, 255, 0)
                            line_thick = max(4, int(out_w / 180))
                            cv2.line(annotated, tuple(waist_center), tuple(f_knee), line_color, line_thick, cv2.LINE_AA)
                            cv2.line(annotated, tuple(f_knee), tuple(f_heel), line_color, line_thick, cv2.LINE_AA)
                            cv2.circle(annotated, tuple(waist_center), 8, (255, 0, 255), -1) 
                            cv2.circle(annotated, tuple(f_knee), 8, (0, 255, 255), -1)      
                            cv2.circle(annotated, tuple(f_heel), 8, (0, 255, 0), -1)        
                            cv2.putText(annotated, f"{current_angle:.1f} deg", (f_knee[0] + 20, f_knee[1]), 
                                        cv2.FONT_HERSHEY_SIMPLEX, max(0.8, out_w/900), line_color, 3)

                    # =========================================================
                    # 🚨 [2단계] 플라잉(Loss of Contact): 원본 이미지 박제
                    # =========================================================
                    flight_gap = global_ground_y - current_lowest_y
                    if flight_gap > (out_h * 0.02): 
                        flight_frames_count += 1
                    else:
                        if flight_frames_count >= required_flight_frames:
                            flight_foul_frames.append(clean_frame)
                        flight_frames_count = 0
                
                # 기본 4배속 슬로우 모션 기록 (3초 정지 구간이 아닐 때만 실행)
                if draw_normal_slowmo:
                    bgr_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    for _ in range(4):
                        out_video.write(bgr_frame)
                        
                # 쿨다운 감소
                if freeze_cooldown > 0:
                    freeze_cooldown -= 1

        cap.release()
        out_video.release()
        
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 최종 리포트 출력
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다.")
    else:
        st.divider()
        st.header("🎬 Rule 54 방송용 하이라이트 VAR")
        
        # --- 1. 무릎 굽힘 (슬로우 모션 영상 재생) ---
        st.subheader("🔴 1. 수직 신전(Bent Knee) 3초 정지 하이라이트")
        st.info("💡 각도가 170도 미만으로 붕괴되는 순간, 화면이 3초간 일시 정지되며 선이 진한 붉은색으로 강조됩니다.")
        
        # 비디오 재생
        st.video(out_video_path)
        
        if worst_bent_angle < 170.0:
            st.error(f"🚨 **파울 감지됨:** 시뮬레이션 구간 중 최저 각도가 **{worst_bent_angle:.1f}°**까지 무너졌습니다.")
        else:
            st.success(f"✅ **통과:** 전 구간 최저 각도가 **{worst_bent_angle:.1f}°**로, 170도 이상을 훌륭하게 유지했습니다.")
            
        st.write("---")
        
        # --- 2. 플라잉 파울 (원본 이미지만 박제) ---
        st.subheader("🟡 2. 플라잉 파울 (두 발 체공 원본 추출)")
        if len(flight_foul_frames) > 0:
            st.warning("⚠️ 두 발이 허공에 0.08초를 초과하여 떠 있는 '플라잉' 장면 원본입니다. (보조선 없음)")
            for i in range(0, len(flight_foul_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(flight_foul_frames):
                        with cols[j]:
                            st.image(flight_foul_frames[i + j], channels="RGB", caption=f"플라잉 포착 원본 #{i+j+1}")
        else:
            st.success("✅ 통과: 두 발이 동시에 0.08초를 초과하여 떠 있는 플라잉 파울이 감지되지 않았습니다.")

st.write("---")
st.info("💡 **방송급 연출 탑재:** 심판진의 명확한 판독을 위해 파울 발생 찰나의 순간을 캐치하여 화면을 멈추고 굵은 시각 효과를 줍니다.")

try:
    os.unlink(out_video_path)
except:
    pass
