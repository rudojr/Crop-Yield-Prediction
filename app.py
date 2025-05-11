import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns


# Hàm tải mô hình từ code đã được cung cấp
def load_models(crop):
    safe_crop = crop.replace(" ", "_").replace("/", "_")
    crop_dir = os.path.join("save_models", safe_crop)  # Đảm bảo đúng tên thư mục (saved_models)

    lstm_model_path = os.path.join(crop_dir, "lstm_model.keras")
    lstm_embedding_path = os.path.join(crop_dir, "lstm_embedding_model.keras")
    xgb_model_path = os.path.join(crop_dir, "xgb_model.pkl")
    scaler_path = os.path.join(crop_dir, "scaler.save")

    if not all(os.path.exists(p) for p in [lstm_model_path, lstm_embedding_path, xgb_model_path, scaler_path]):
        return None, None, None, None

    lstm_model = tf.keras.models.load_model(lstm_model_path)
    lstm_embedding_model = tf.keras.models.load_model(lstm_embedding_path)
    xgb_model = joblib.load(xgb_model_path)
    scaler = joblib.load(scaler_path)

    return lstm_model, lstm_embedding_model, xgb_model, scaler


# Hàm dự đoán năng suất
def predict_yield(crop, state, season, area, rainfall, fertilizer, pesticide, lstm_embedding_model, xgb_model, scaler):
    # Tạo dữ liệu đầu vào
    # Đầu tiên chuẩn bị dữ liệu chuỗi thời gian (3 năm gần nhất)
    seq_data = np.array([[rainfall, fertilizer, pesticide]] * 3).reshape(1, 3, 3)

    # Chuẩn hóa dữ liệu chuỗi thời gian
    seq_data_flat = seq_data.reshape(-1, seq_data.shape[-1])
    seq_data_scaled = scaler.transform(seq_data_flat).reshape(seq_data.shape)

    # Tạo đặc trưng tĩnh
    # Lấy dữ liệu để tạo One-hot encoding cho State và Season
    df = pd.read_csv("data.csv")  # Đọc file dữ liệu gốc
    df_sample = df[df['Crop'] == crop].copy()
    df_sample = pd.get_dummies(df_sample, columns=['Season', 'State'], drop_first=True)

    # Lấy tất cả các cột State_ và Season_
    state_cols = [col for col in df_sample.columns if col.startswith('State_')]
    season_cols = [col for col in df_sample.columns if col.startswith('Season_')]

    # Tạo đặc trưng tĩnh
    static_data = np.zeros(len(state_cols) + len(season_cols) + 1)  # +1 cho Area

    # Đặt giá trị cho Area
    static_data[-1] = area

    # Đặt giá trị One-hot cho State
    for i, col in enumerate(state_cols):
        state_name = col.replace('State_', '')
        if state_name == state:
            static_data[i] = 1
            break

    # Đặt giá trị One-hot cho Season
    offset = len(state_cols)
    for i, col in enumerate(season_cols):
        season_name = col.replace('Season_', '')
        if season_name == season:
            static_data[offset + i] = 1
            break

    # Sử dụng mô hình LSTM để tạo embedding
    lstm_embedding = lstm_embedding_model.predict(seq_data_scaled)

    # Kết hợp embedding và đặc trưng tĩnh
    xgb_input = np.concatenate([lstm_embedding, static_data.reshape(1, -1)], axis=1)

    # Dự đoán với XGBoost
    prediction = xgb_model.predict(xgb_input)[0]

    return prediction


# Hàm tải dữ liệu lịch sử năng suất cho biểu đồ
def load_historical_data(crop, state, season):
    df = pd.read_csv("data.csv")
    filtered_df = df[(df['Crop'] == crop) & (df['State'] == state) & (df['Season'] == season)].sort_values('Crop_Year')
    return filtered_df


# Giao diện chính
st.title("🌾 Dự đoán năng suất nông sản (Yield Prediction)")

# Tải dữ liệu
try:
    df = pd.read_csv("data.csv")
    crop_list = df["Crop"].unique()
    season_list = df["Season"].unique()
    state_list = df["State"].unique()
except Exception as e:
    st.error(f"Lỗi khi tải dữ liệu: {e}")
    st.stop()

# Layout chính
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Thông số đầu vào")

    crop_name = st.selectbox("Chọn loại cây trồng (Crop)", crop_list)
    lstm_model, embedding_model, xgb_model, scaler = load_models(crop_name)

    if lstm_model is None:
        st.error(f"Không tìm thấy mô hình cho {crop_name}. Vui lòng chọn loại cây trồng khác.")
        st.stop()

    state = st.selectbox("Chọn Bang/Thành phố (State)", state_list)
    season = st.selectbox("Chọn Mùa vụ (Season)", season_list)

    area = st.number_input("Diện tích gieo trồng (Area - ha)", min_value=0.0, value=1.0)
    annual_rainfall = st.number_input("Lượng mưa trung bình hàng năm (mm)", min_value=0.0, value=800.0)
    fertilizer = st.number_input("Lượng phân bón (kg)", min_value=0.0, value=50.0)
    pesticide = st.number_input("Lượng thuốc trừ sâu (kg)", min_value=0.0, value=10.0)

    if st.button("Dự đoán"):
        with st.spinner('Đang tính toán...'):
            try:
                # Thực hiện dự đoán
                prediction = predict_yield(
                    crop_name, state, season, area, annual_rainfall,
                    fertilizer, pesticide, embedding_model, xgb_model, scaler
                )

                # Hiển thị kết quả
                st.success(f"Năng suất dự đoán: {prediction:.2f} tấn/ha")

                # Hiển thị dữ liệu lịch sử
                st.subheader("So sánh với dữ liệu lịch sử")
                historical_data = load_historical_data(crop_name, state, season)

                if not historical_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    years = historical_data['Crop_Year'].tolist()
                    yields = historical_data['Yield'].tolist()

                    # Vẽ dữ liệu lịch sử
                    ax.plot(years, yields, marker='o', linestyle='-', color='blue', label='Năng suất lịch sử')

                    # Thêm điểm dự đoán
                    current_year = max(years) + 1 if years else 2025
                    ax.scatter([current_year], [prediction], color='red', s=100, label='Dự đoán')
                    ax.plot([years[-1], current_year], [yields[-1], prediction], 'r--')

                    ax.set_xlabel('Năm')
                    ax.set_ylabel('Năng suất (tấn/ha)')
                    ax.set_title(f'Năng suất {crop_name} tại {state} vào mùa {season}')
                    ax.grid(True)
                    ax.legend()

                    st.pyplot(fig)
                else:
                    st.info("Không có dữ liệu lịch sử cho sự kết hợp này.")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")

with col2:
    st.subheader("Thông tin mô hình")
    st.info(f"""
    Thông tin về mô hình dự đoán cho {crop_name}:

    - Sử dụng mô hình lai kết hợp giữa LSTM và XGBoost
    - LSTM xử lý dữ liệu chuỗi thời gian (3 năm)
    - XGBoost kết hợp đặc trưng chuỗi thời gian và đặc trưng tĩnh
    """)

    st.subheader("Hướng dẫn sử dụng")
    st.info("""
    1. Chọn loại cây trồng cần dự đoán
    2. Chọn Vùng/Bang và Mùa vụ
    3. Nhập diện tích canh tác
    4. Nhập thông số về lượng mưa, phân bón và thuốc trừ sâu
    5. Nhấn nút Dự đoán để xem kết quả
    """)