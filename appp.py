import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from playsound import playsound
import joblib
import sqlite3
from datetime import datetime

# === Custom CSS ===
st.set_page_config(page_title="Bicep Curl Monitor", layout="centered")
st.markdown("""
    <style>
    .stButton>button {
        background-color: #28a745;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-weight: bold;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 8px;
        padding: 0.5em;
        border: 1px solid #ccc;
    }
    .metric-box {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# === Setup Database ===
def create_db():
    conn = sqlite3.connect('bicep_curl.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS hasil_latihan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama TEXT,
            tanggal TEXT,
            target_reps INTEGER,
            total_reps INTEGER,
            benar INTEGER,
            salah INTEGER,
            akurasi REAL
        )
    ''')
    conn.commit()
    conn.close()

def simpan_data(nama, target, total, benar, salah, akurasi):
    conn = sqlite3.connect('bicep_curl.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO hasil_latihan (nama, tanggal, target_reps, total_reps, benar, salah, akurasi)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (nama, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), target, total, benar, salah, akurasi))
    conn.commit()
    conn.close()

create_db()

# === Load Model ===
knn = joblib.load('model_knn_bicep.pkl')

# === Inisialisasi MediaPipe ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# === Header UI ===
st.title("💪 Sistem Monitoring Bicep Curl")
st.markdown("Selamat datang di aplikasi monitoring bicep curl! Isi nama dan pilih target repetisi.")

# === Input Form ===
with st.container():
    st.subheader("📋 Data Pengguna")
    col1, col2 = st.columns(2)
    with col1:
        user_name = st.text_input("👤 Masukkan nama Anda")
    with col2:
        target_reps = st.selectbox("🎯 Pilih target repetisi", [10, 20, 30, 40, 50])

# === Mulai Kamera ===
if st.button("▶️ Mulai Latihan"):
    if not user_name.strip():
        st.warning("⚠️ Silakan masukkan nama terlebih dahulu.")
    else:
        counter_left = counter_right = 0
        correct_left = correct_right = 0
        incorrect_left = incorrect_right = 0
        stage_left = stage_right = None
        target_reached = False

        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and not target_reached:
                ret, frame = cap.read()
                if not ret:
                    st.error("Webcam tidak dapat diakses.")
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark

                    # Kiri
                    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle_left = calculate_angle(shoulder_l, elbow_l, wrist_l)

                    # Kanan
                    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    angle_right = calculate_angle(shoulder_r, elbow_r, wrist_r)

                    # Logika Lengan Kiri
                    if 160 <= angle_left <= 175:
                        stage_left = "down"
                    if angle_left <= 25 and stage_left == "down":
                        stage_left = "up"
                        counter_left += 1
                        if 10 <= angle_left <= 25:
                            correct_left += 1
                        else:
                            incorrect_left += 1

                    # Logika Lengan Kanan
                    if 160 <= angle_right <= 175:
                        stage_right = "down"
                    if angle_right <= 25 and stage_right == "down":
                        stage_right = "up"
                        counter_right += 1
                        if 10 <= angle_right <= 25:
                            correct_right += 1
                        else:
                            incorrect_right += 1

                except:
                    pass

                total_reps = counter_left + counter_right
                if total_reps >= target_reps:
                    if not target_reached:
                        try:
                            playsound('notif.mp3')
                        except Exception as e:
                            print(f"[Warning] Gagal memutar suara: {e}")
                        target_reached = True

                progress_bar.progress(min(total_reps / target_reps, 1.0))
                cv2.rectangle(image, (0, 0), (400, 100), (0, 0, 0), -1)
                cv2.putText(image, f'Nama : {user_name}', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
                cv2.putText(image, f'Kiri : {counter_left} | Kanan : {counter_right}', (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, f'Benar : {correct_left + correct_right}', (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(image, f'Salah : {incorrect_left + incorrect_right}', (200, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                frame_placeholder.image(image, channels="BGR", use_container_width=True)

            cap.release()

        # === Ringkasan ===
        total_correct = correct_left + correct_right
        total_incorrect = incorrect_left + incorrect_right
        total_detected = total_correct + total_incorrect
        accuracy = (total_correct / total_detected) * 100 if total_detected > 0 else 0
        simpan_data(user_name, target_reps, total_reps, total_correct, total_incorrect, accuracy)

        st.subheader("📊 Ringkasan Hasil Latihan")

        conn = sqlite3.connect('bicep_curl.db')
        query = f"""
        SELECT nama, target_reps, total_reps, benar, salah, akurasi
        FROM hasil_latihan
        WHERE nama = '{user_name}'
        ORDER BY tanggal DESC
        LIMIT 1
        """
        result = pd.read_sql_query(query, conn)
        conn.close()

        if result.empty:
            st.warning("⚠️ Data latihan tidak ditemukan.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='metric-box'><b>👤 Nama :</b> {result.iloc[0]['nama']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'><b>🎯 Target :</b> {result.iloc[0]['target_reps']} Reps</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'><b>🔁 Total Reps :</b> {result.iloc[0]['total_reps']}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-box'><b>✅ Benar :</b> {result.iloc[0]['benar']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'><b>❌ Salah :</b> {result.iloc[0]['salah']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'><b>📈 Akurasi :</b> {result.iloc[0]['akurasi']:.2f}%</div>", unsafe_allow_html=True)

# === Footer ===
st.markdown("""
<hr>
<p style="text-align: center; font-size: 0.8rem; color: #aaa;">© 2025 Sistem Monitoring Bicep Curl | Dibuat </p>
""", unsafe_allow_html=True)
