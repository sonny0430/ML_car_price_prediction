# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# 기본 설정
# ---------------------------------------------------
st.set_page_config(
    page_title="중고차 가격 예측",
    page_icon="🚗",
    layout="wide"
)

INR_TO_KRW = 16.0

# ---------------------------------------------------
# 최소 CSS (과한 전역 스타일 제거)
# ---------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# 모델 패키지 로드
# ---------------------------------------------------
package = joblib.load("model/car_price_model_package.pkl")
model = package["model"]
model_columns = package["columns"]
owner_map = package["owner_map"]
top_brands = package["top_brands"]
reference_year = package["reference_year"]

# ---------------------------------------------------
# 모델 성능 지표
# ---------------------------------------------------
MODEL_NAME = "LightGBM"
TRAIN_R2 = 0.9473
TEST_R2 = 0.9124
MAE = 70307.21
RMSE = 128015.27
R2_ORIGINAL = 0.9253

# ---------------------------------------------------
# 시각화용 데이터 로드
# ---------------------------------------------------
df_vis = pd.read_csv("dataset/Car details v3.csv")
df_vis = df_vis.drop_duplicates().reset_index(drop=True)

def extract_first_number(x):
    if x is None or pd.isna(x):
        return np.nan
    match = re.search(r'(\d+\.?\d*)', str(x))
    if match:
        return float(match.group(1))
    return np.nan

df_vis["brand_name"] = df_vis["name"].apply(lambda x: str(x).split()[0])
df_vis["car_age"] = reference_year - df_vis["year"]
df_vis["selling_price"] = pd.to_numeric(df_vis["selling_price"], errors="coerce")
df_vis["float_max_power"] = df_vis["max_power"].apply(extract_first_number)
df_vis = df_vis.dropna(subset=["selling_price", "car_age", "float_max_power"])

# ---------------------------------------------------
# 전처리 함수
# ---------------------------------------------------
def preprocess_input(input_df):
    df = input_df.copy()

    df["car_age"] = reference_year - df["year"]
    df["float_mileage"] = df["mileage"].apply(extract_first_number)
    df["float_engine"] = df["engine"].apply(extract_first_number)
    df["float_max_power"] = df["max_power"].apply(extract_first_number)
    df["float_torque"] = df["torque"].apply(extract_first_number)

    df["owner_encoding"] = df["owner"].map(owner_map)
    df["top_brand"] = df["brand_name"].apply(lambda x: x if x in top_brands else "others")

    df = pd.get_dummies(
        df,
        columns=["fuel", "seller_type", "transmission", "top_brand"],
        drop_first=True
    )

    use_cols = [
        "car_age",
        "km_driven",
        "seats",
        "float_mileage",
        "float_engine",
        "float_max_power",
        "float_torque",
        "owner_encoding"
    ]

    base_df = df[use_cols].copy()
    dummy_cols = [col for col in df.columns if col not in input_df.columns and col not in use_cols]
    final_df = pd.concat([base_df, df[dummy_cols]], axis=1)
    final_df = final_df.reindex(columns=model_columns, fill_value=0)

    return final_df

# ---------------------------------------------------
# 상단 소개
# ---------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1f77b4, #4a90e2);
    padding: 1.5rem 1.7rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1.2rem;
">
    <h1 style="color:white; margin:0 0 0.4rem 0;">🚗 AI 기반 중고차 가격 예측</h1>
    <p style="color:white; margin:0;">
        차량 정보를 입력하면 예상 중고차 가격을 예측합니다.<br>
        본 모델은 인도 중고차 데이터를 기반으로 학습되었으며, 결과는 <b>원화(KRW)</b> 기준으로 환산하여 제공합니다.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📌 가격 예측", "📊 모델 정보"])

# ===================================================
# TAB 1 : 가격 예측
# ===================================================
with tab1:
    st.subheader("차량 정보 입력")
    st.caption("※ 기준 연도는 2020년으로 설정되어 있으며, 연식 입력은 2020년 이하로 제한됩니다.")

    left, right = st.columns(2)

    with left:
        year = st.number_input(
            "연식 (Year)",
            min_value=1990,
            max_value=reference_year,
            value=2018,
            step=1
        )

        km_driven = st.number_input(
            "주행거리 (km)",
            min_value=0,
            max_value=500000,
            value=40000,
            step=1000
        )

        seats = st.number_input(
            "좌석 수",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=1.0
        )

        fuel = st.selectbox(
            "연료 타입",
            ["Diesel", "Petrol", "CNG", "LPG"]
        )

        seller_type = st.selectbox(
            "판매자 유형",
            ["Individual", "Dealer", "Trustmark Dealer"]
        )

        transmission = st.selectbox(
            "변속기",
            ["Manual", "Automatic"]
        )

    with right:
        owner = st.selectbox(
            "소유주 이력",
            list(owner_map.keys())
        )

        brand_name = st.selectbox(
            "브랜드",
            top_brands + ["others"]
        )

        float_mileage = st.number_input(
            "연비 (kmpl)",
            min_value=0.0,
            max_value=50.0,
            value=18.0,
            step=0.1
        )

        float_engine = st.number_input(
            "배기량 (CC)",
            min_value=500.0,
            max_value=5000.0,
            value=1200.0,
            step=10.0
        )

        float_max_power = st.number_input(
            "최대출력 (bhp)",
            min_value=20.0,
            max_value=500.0,
            value=80.0,
            step=1.0
        )

        float_torque = st.number_input(
            "토크 (Nm)",
            min_value=0.0,
            max_value=1000.0,
            value=120.0,
            step=1.0
        )

    input_df = pd.DataFrame([{
        "year": year,
        "km_driven": km_driven,
        "seats": seats,
        "mileage": f"{float_mileage} kmpl",
        "engine": f"{float_engine} CC",
        "max_power": f"{float_max_power} bhp",
        "torque": f"{float_torque} Nm",
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "brand_name": brand_name
    }])

    st.markdown("---")

    if st.button("💰 예상 가격 예측"):
        X_input = preprocess_input(input_df)
        pred_log = model.predict(X_input)[0]

        pred_inr = np.expm1(pred_log)
        pred_krw = pred_inr * INR_TO_KRW
        input_car_age = reference_year - year

        c1, c2 = st.columns(2)
        c1.metric("예상 가격 (KRW)", f"₩ {pred_krw:,.0f}")
        c2.metric("참고 가격 (INR)", f"₹ {pred_inr:,.0f}")

        avg_price = df_vis["selling_price"].mean()
        c3, c4 = st.columns(2)
        c3.metric("전체 평균 가격 (INR)", f"₹ {avg_price:,.0f}")
        c4.metric("평균 대비 차이", f"₹ {pred_inr - avg_price:,.0f}")

        st.markdown("""
        <div style="
            background:#eef5ff;
            border-left:6px solid #4a90e2;
            border-radius:14px;
            padding:1rem 1.2rem;
            margin-top:1rem;
            margin-bottom:1rem;
            color:#1f1f1f;
        ">
            본 예측 결과는 참고용 값이며, 실제 거래 가격은 차량 상태, 옵션, 지역, 시장 상황 등에 따라 달라질 수 있습니다.
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # 시각화 1: 전체 가격 분포 + 예측값
        # -------------------------
        st.subheader("예측 가격의 전체 분포상 위치")

        fig1, ax1 = plt.subplots(figsize=(9, 4))
        sns.histplot(df_vis["selling_price"], bins=40, kde=True, ax=ax1, color="#7aa6d8")
        ax1.axvline(pred_inr, color="red", linestyle="--", linewidth=2, label="Predicted Price")
        ax1.set_title("Distribution of Selling Price")
        ax1.set_xlabel("Selling Price (INR)")
        ax1.legend()
        st.pyplot(fig1)

        # -------------------------
        # 시각화 2: car_age vs price + 입력값
        # -------------------------
        st.subheader("입력 차량의 감가상각 구조상 위치")

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            data=df_vis,
            x="car_age",
            y="selling_price",
            alpha=0.25,
            ax=ax2,
            color="#5b8ecb"
        )
        ax2.scatter(
            input_car_age,
            pred_inr,
            color="red",
            s=180,
            marker="*",
            label="Your Input"
        )
        ax2.set_title("Car Age vs Selling Price")
        ax2.set_xlabel("Car Age")
        ax2.set_ylabel("Selling Price (INR)")
        ax2.legend()
        st.pyplot(fig2)

        # -------------------------
        # 시각화 3: max_power vs price + 입력값
        # -------------------------
        st.subheader("입력 차량의 성능-가격 구조상 위치")

        fig3, ax3 = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            data=df_vis,
            x="float_max_power",
            y="selling_price",
            alpha=0.25,
            ax=ax3,
            color="#74a9cf"
        )
        ax3.scatter(
            float_max_power,
            pred_inr,
            color="red",
            s=180,
            marker="*",
            label="Your Input"
        )
        ax3.set_title("Max Power vs Selling Price")
        ax3.set_xlabel("Max Power (bhp)")
        ax3.set_ylabel("Selling Price (INR)")
        ax3.legend()
        st.pyplot(fig3)

        with st.expander("입력값 요약 보기"):
            st.write({
                "연식": year,
                "주행거리(km)": km_driven,
                "좌석 수": seats,
                "연비(kmpl)": float_mileage,
                "배기량(CC)": float_engine,
                "최대출력(bhp)": float_max_power,
                "토크(Nm)": float_torque,
                "연료": fuel,
                "판매자 유형": seller_type,
                "변속기": transmission,
                "소유주 이력": owner,
                "브랜드": brand_name
            })

# ===================================================
# TAB 2 : 모델 정보
# ===================================================
with tab2:
    st.subheader("최종 모델 성능 정보")

    m1, m2, m3 = st.columns(3)
    m1.metric("최종 모델", MODEL_NAME)
    m2.metric("Train R²", f"{TRAIN_R2:.4f}")
    m3.metric("Test R²", f"{TEST_R2:.4f}")

    m4, m5, m6 = st.columns(3)
    m4.metric("MAE", f"{MAE:,.0f}")
    m5.metric("RMSE", f"{RMSE:,.0f}")
    m6.metric("R² (Original Scale)", f"{R2_ORIGINAL:.4f}")

    st.markdown("---")

    st.markdown("""
    <div style="
        background:white;
        border-left:6px solid #1f77b4;
        border-radius:14px;
        padding:1rem 1.2rem;
        margin-bottom:1rem;
        color:#1f1f1f;
    ">
        <h4 style="color:#1f3b73; margin-top:0;">모델 설명</h4>
        <ul style="color:#1f1f1f;">
            <li>최종 모델은 <b>LightGBM 기반 회귀모델</b>입니다.</li>
            <li>Target 변수는 <b>log_selling_price</b>로 학습하였으며, 예측 후 원래 가격 스케일로 복원하였습니다.</li>
            <li>주요 입력 변수는 <b>car_age, km_driven, seats, mileage, engine, max_power, torque, owner, fuel, seller_type, transmission, brand</b> 입니다.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:white;
        border-left:6px solid #1f77b4;
        border-radius:14px;
        padding:1rem 1.2rem;
        margin-bottom:1rem;
        color:#1f1f1f;
    ">
        <h4 style="color:#1f3b73; margin-top:0;">해석 참고</h4>
        <ul style="color:#1f1f1f;">
            <li>본 모델은 연식 기반 감가상각 효과와 차량 성능 변수의 영향을 크게 반영합니다.</li>
            <li>브랜드 및 거래 유형도 반영되지만, 연식과 성능 변수의 영향이 상대적으로 더 크게 나타날 수 있습니다.</li>
            <li>기준 연도(reference year)는 <b>2020년</b>입니다.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("추가 정보 보기"):
        st.write(f"최종 입력 feature 수: {len(model_columns)}개")
        st.write(f"상위 브랜드 수: {len(top_brands)}개")
        st.write(f"기준 연도(reference year): {reference_year}")