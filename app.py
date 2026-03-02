import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Rule 54 Video VAR", layout="wide")
st.title("🎬 Rule 54 공식 VAR (영상 시뮬레이션 모드)")
st.markdown("##### 💡 1. 무릎 굽힘(Bent Knee): AI가 '종골-무릎중앙-몸통수직축' 각도 변화를 프레임 단위로 추적하는 '시뮬레이션 영상'을 제공합니다.")
st.markdown("##### 💡 2. 플라잉(Loss of Contact): 두 발이 떠 있는 순간은 보조선이 없는 '원본 사진'으로만 박제합니다.")
st.write("---")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    # 몸통 수직축(a) - 무릎 중심(b) - 종골(c) 사이의 각도
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

st.error("⚠️ **10초 이내의 경보 영상**을 올려주세요. 시뮬레이션 추적 영상을 생성합니다.")
video_file = st.file_uploader("경보 영상 업로드 (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if video_file:
    # 입력 영상 임시 저장
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close() 
    
    # 💡 출력 시뮬레이션 영상 임시 저장 (웹에서 재생 잘 되는 webm 형식 사용)
    out_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
    
    flight_foul_frames = [] 
    
    global_ground_y = 0.0
    flight_frames_count = 0 
    
    worst_bent_angle = 180.0 
    person_detected = False

    try:
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0 
        
        # 원본 영상 해상도 가져오기
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 연산 및 출력을 위해 가로 800픽셀로 스케일링
        scale_ratio = 800 / orig_w if orig_w > 800 else 1.0
        out_w = int(orig_w * scale_ratio)
        out_h = int(orig_h * scale_ratio)
        
        # VideoWriter 초기화 (VP8 코덱이 웹 브라우저 호환성이 가장 좋습니다)
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, out_h))
            
        required_flight_frames = int(0.08 * fps)
        if required_flight_frames < 2: required_flight_frames = 2

        with st.spinner("🕵️‍♂️ AI가 각도 변화를 프레임 단위로 추적하는 시뮬레이션 영상을 제작 중입니다..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # 프레임 리사이즈
                if scale_ratio != 1.0:
                    frame = cv2.resize(frame, (out_w, out_h))
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                
                # 💡 플라잉 추출용 원본(clean) 캡처 (선 긋기 전)
                clean_frame = img.copy()
                annotated = img.copy() # 시뮬레이션 영상에 쓰일 도화지
                
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
                    # 🚨 [1단계] 영상 시뮬레이션: 종골 착지 시 각도 변화 실시간 묘사
                    # =========================================================
                    f_heel = l_heel if front_is_left else r_heel
                    f_knee = l_k if front_is_left else r_k
                    
                    is_in_front = (f_heel[0] > waist_center[0]) if moving_right else (f_heel[0] < waist_center[0])
                    is_grounded = abs(global_ground_y - f_heel[1]) < (out_h * 0.04)

                    if is_in_front and is_grounded:
                        current_angle = calculate_angle(waist_center, f_knee, f_heel)
                        
                        if current_angle < worst_bent_angle:
                            worst_bent_angle = current_angle
                            
                        # 💡 영상에 실시간 각도 변화 렌더링 (170도 미만이면 빨간색, 이상이면 초록색)
                        line_color = (0, 0, 255) if current_angle < 170.0 else (0, 255, 0)
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
                            # 💡 각도나 선이 없는 'clean_frame' 원본만 저장
                            flight_foul_frames.append(clean_frame)
                        flight_frames_count = 0

                # 시뮬레이션 화면을 비디오 파일로 기록 (OpenCV는 BGR을 쓰므로 다시 변환)
                out_video.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

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
        st.header("🎬 Rule 54 AI 시뮬레이션 VAR 리포트")
        
        # --- 1. 무릎 굽힘 (영상 재생) ---
        st.subheader("🔴 1. 수직 신전(Bent Knee) 추적 영상")
        st.info("💡 AI가 종골 착지부터 수직 구간까지의 각도를 1프레임 단위로 추적한 시뮬레이션 영상입니다. (170도 미만 시 빨간선)")
        
        # 💡 생성된 시뮬레이션 비디오 재생
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
                        # 깨끗한 원본 이미지 출력
                        with cols[j]:
                            st.image(flight_foul_frames[i + j], channels="RGB", caption=f"플라잉 포착 원본 #{i+j+1}")
        else:
            st.success("✅ 통과: 두 발이 동시에 0.08초를 초과하여 떠 있는 플라잉 파울이 감지되지 않았습니다.")

st.write("---")
st.info("💡 **판독 방식:** 무릎 신전 여부는 사용자가 직접 눈으로 확인할 수 있도록 프레임 단위의 **'추적 영상'**으로 변환하여 보여주며, 플라잉 장면은 방해되는 선 없이 100% **'원본 깨끗한 사진'**만 캡처하여 제공합니다.")

# 처리가 끝난 후 임시 비디오 파일은 찌꺼기 방지를 위해 삭제 시도 (선택)
try:
    os.unlink(out_video_path)
except:
    pass
