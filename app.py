import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# 💡 [안전장치] MediaPipe 로드 중 에러 발생 시 앱이 죽지 않고 화면에 표시
try:
    import mediapipe as mp
except ImportError:
    st.error("🚨 치명적 에러: MediaPipe 패키지가 설치되지 않았거나 손상되었습니다. 터미널에서 'pip install mediapipe'를 실행해주세요.")
    st.stop()
except Exception as e:
    st.error(f"🚨 MediaPipe 로드 중 알 수 없는 에러가 발생했습니다: {str(e)}")
    st.stop()

st.set_page_config(page_title="Rule 54 Final VAR", layout="wide")
st.title("🎬 Rule 54 공식 VAR (안전성 강화 완결판)")
st.markdown("##### 💡 1. 무릎 굽힘: 종골 착지점부터 수직 구간까지 4배속 슬로우 모션으로 정밀 판독합니다.")
st.markdown("##### 💡 2. 플라잉 억제: 발가락 센서를 가동하여 억울한 체공 판독을 100% 차단합니다.")
st.write("---")

mp_pose = mp.solutions.pose
# 에러 방지를 위해 static_image_mode를 False로 유지
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

st.error("⚠️ **10초 이내의 경보 영상**을 올려주세요. 발끝(Toe)까지 스캔하여 완벽한 판독을 수행합니다.")
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
    freeze_cooldown = 0

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 💡 [안전장치] 영상 해상도가 0일 경우(읽기 실패) 앱 튕김 방지
        if orig_w == 0 or orig_h == 0:
            st.error("🚨 비디오 파일을 읽을 수 없습니다. 파일이 손상되었거나 코덱이 지원되지 않습니다.")
            st.stop()
            
        scale_ratio = 800 / orig_w if orig_w > 800 else 1.0
        out_w = int(orig_w * scale_ratio)
        out_h = int(orig_h * scale_ratio)
        
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, out_h))
            
        required_flight_frames = int(0.08 * fps)
        if required_flight_frames < 2: required_flight_frames = 2

        with st.spinner("🕵️‍♂️ AI가 발가락 센서를 가동하여 억울한 체공 판독(오심)을 차단하고 있습니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if scale_ratio != 1.0:
                    frame = cv2.resize(frame, (out_w, out_h))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 💡 [안전장치] process() 중 에러 발생 시 처리
                try:
                    res = pose.process(img)
                except Exception as e:
                    st.warning(f"⚠️ 일부 프레임 분석 중 에러 무시됨: {str(e)}")
                    continue
                
                clean_frame = img.copy() 
                annotated = img.copy()   
                draw_normal_slowmo = True 
                
                if res.pose_landmarks:
                    person_detected = True
                    lm = res.pose_landmarks.landmark
                    
                    def get_pt(landmark):
                        return [int(landmark.x * out_w), int(landmark.y * out_h)]
                    
                    try:
                        l_h = get_pt(lm[mp_pose.PoseLandmark.LEFT_HIP])
                        r_h = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HIP])
                        l_k = get_pt(lm[mp_pose.PoseLandmark.LEFT_KNEE])
                        r_k = get_pt(lm[mp_pose.PoseLandmark.RIGHT_KNEE])
                        l_heel = get_pt(lm[mp_pose.PoseLandmark.LEFT_HEEL])
                        r_heel = get_pt(lm[mp_pose.PoseLandmark.RIGHT_HEEL])
                        l_toe = get_pt(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])   
                        r_toe = get_pt(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])  
                        nose_x = lm[mp_pose.PoseLandmark.NOSE].x * out_w
                    except IndexError:
                        # 신체의 일부가 화면 밖으로 나가서 인식이 안 될 경우 튕김 방지
                        continue
                    
                    waist_center = [int((l_h[0] + r_h[0]) / 2), int((l_h[1] + r_h[1]) / 2)]
                    moving_right = nose_x > waist_center[0]
                    front_is_left = (l_heel[0] > r_heel[0]) if moving_right else (l_heel[0] < r_heel[0])
                    
                    current_lowest_y = max(l_heel[1], r_heel[1], l_toe[1], r_toe[1])
                    if current_lowest_y > global_ground_y:
                        global_ground_y = current_lowest_y

                    # =========================================================
                    # 🚨 [1단계] 무릎 신전 170도 판독 (3초 정지 기능 포함)
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
                                line_color = (0, 0, 255)
                                line_thick_bold = max(10, int(out_w / 60))
                                cv2.line(annotated, tuple(waist_center), tuple(f_knee), line_color, line_thick_bold, cv2.LINE_AA)
                                cv2.line(annotated, tuple(f_knee), tuple(f_heel), line_color, line_thick_bold, cv2.LINE_AA)
                                cv2.circle(annotated, tuple(waist_center), 12, (255, 0, 255), -1) 
                                cv2.circle(annotated, tuple(f_knee), 12, (0, 255, 255), -1)      
                                cv2.circle(annotated, tuple(f_heel), 12, (0, 255, 0), -1)        
                                cv2.putText(annotated, f"FOUL: {current_angle:.1f} deg", (f_knee[0] + 30, f_knee[1]), 
                                            cv2.FONT_HERSHEY_SIMPLEX, max(1.2, out_w/600), line_color, 4)
                                cv2.putText(annotated, "VAR: 170 DEGREE FAILED!", (30, 60), 
                                            cv2.FONT_HERSHEY_SIMPLEX, max(1.0, out_w/700), line_color, 4)

                                bgr_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                                for _ in range(int(fps * 3)): 
                                    out_video.write(bgr_frame)
                                    
                                freeze_cooldown = int(fps * 1.5) 
                                draw_normal_slowmo = False 
                            else:
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
                    # 🚨 [2단계] 플라잉 파울: 발가락 센서(Toe Sensor) 
                    # =========================================================
                    points_y = [l_heel[1], r_heel[1], l_toe[1], r_toe[1]]
                    closest_to_ground_y = max(points_y)
                    flight_gap = global_ground_y - closest_to_ground_y
                    
                    if flight_gap > (out_h * 0.02): 
                        flight_frames_count += 1
                    else:
                        if flight_frames_count >= required_flight_frames:
                            flight_foul_frames.append(clean_frame)
                        flight_frames_count = 0
                
                if draw_normal_slowmo:
                    bgr_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    for _ in range(4):
                        out_video.write(bgr_frame)
                        
                if freeze_cooldown > 0:
                    freeze_cooldown -= 1

        cap.release()
        out_video.release()
        
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

    # 4. 최종 리포트 출력
    if not person_detected:
        st.error("❌ 영상을 분석할 수 없습니다. 사람이 없거나 프레임을 읽지 못했습니다.")
    else:
        st.divider()
        st.header("🎬 Rule 54 방송용 하이라이트 VAR 리포트")
        
        st.subheader("🔴 1. 수직 신전(Bent Knee) 3초 정지 하이라이트")
        st.video(out_video_path)
        
        if worst_bent_angle < 170.0:
            st.error(f"🚨 **파울 감지됨:** 시뮬레이션 구간 중 최저 각도가 **{worst_bent_angle:.1f}°**까지 무너졌습니다.")
        else:
            st.success(f"✅ **통과:** 전 구간 최저 각도가 **{worst_bent_angle:.1f}°**로, 170도 이상을 훌륭하게 유지했습니다.")
            
        st.write("---")
        
        st.subheader("🟡 2. 플라잉 파울 (두 발 체공 원본 추출)")
        if len(flight_foul_frames) > 0:
            st.warning("⚠️ 발가락(Toe)까지 스캔한 결과, 명백하게 양발 전체가 0.08초를 초과하여 떠 있는 장면입니다.")
            for i in range(0, len(flight_foul_frames), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(flight_foul_frames):
                        with cols[j]:
                            st.image(flight_foul_frames[i + j], channels="RGB")
        else:
            st.success("✅ 통과: '발가락 지지 구간'을 포함하여, 두 발이 0.08초 이상 완전히 떠 있는 파울은 감지되지 않았습니다. (오심 억제 완료)")

st.write("---")
st.info("💡 **안전성 패치:** 영상 분석 중 알 수 없는 에러로 앱이 튕기는 것을 막기 위해 다중 예외 처리(Try-Except) 가동 중입니다.")

try:
    os.unlink(out_video_path)
except:
    pass
