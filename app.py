import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Modelni yuklab olish
model = joblib.load("model.pkl")

st.title("💰 Kredit berish Modeli")
st.markdown("Iltimos, quyidagi ma'lumotlarni to‘ldiring:")

# Foydalanuvchidan inputlar
loan_amount = st.number_input("Kredit summasi ($)", min_value=100.0, max_value=50000.0, step=100.0)
income = st.number_input("Yillik daromad ($)", min_value=1000.0, max_value=200000.0, step=100.0)

term = st.selectbox("Kredit muddati", ["Short Term", "Long Term"])
emp_length = st.selectbox("Ish tajribasi (yil)", list(range(0, 11)))
home_ownership = st.selectbox("Uy egalik turi", ["RENT", "MORTGAGE", "OWN", "OTHER"])
purpose = st.selectbox("Kredit maqsadi", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "car", "wedding", "medical", "vacation", "other"])
dti = st.number_input("Debt-to-Income (DTI)", min_value=0.0, max_value=50.0, step=0.1)
delinq_2yrs = st.number_input("So'nggi 2 yildagi kechikishlar soni", min_value=0, max_value=20, step=1)
revol_util = st.number_input("Revolving utilization (%)", min_value=0.0, max_value=150.0, step=0.1)
total_acc = st.number_input("Umumiy hisoblar soni", min_value=0, max_value=100, step=1)
last_delinquent = st.number_input("Oxirgi kechikishdan beri o'tgan oylar", min_value=0, max_value=120, step=1)

# Tugma
if st.button("✅ Kredit ehtimolini hisobla"):
    # Inputni DataFrame shaklida tayyorlash
    input_data = pd.DataFrame({
        "Loan Amount": [np.log1p(loan_amount)],
        "Annual Income": [np.log1p(income)],
        "Term": [term],
        "Employment Length": [emp_length],
        "Home Ownership": [home_ownership],
        "Purpose": [purpose],
        "Debt-To-Income Ratio": [dti],
        "Delinquent 2yrs": [delinq_2yrs],
        "Revolving Utilization": [revol_util],
        "Total Accounts": [total_acc],
        "Months since last delinquent": [last_delinquent]
    })

    # To‘g‘ri formatlash uchun dummy encoding qilish (modelda qanday bo‘lsa, shunday)
    input_data = pd.get_dummies(input_data)

    # Modelda mavjud bo‘lgan ustunlarni olish
    model_features = model.feature_names_

    # Modelga mos bo‘lmagan ustunlarni to‘ldirish (0 bilan)
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0

    # Modelga mos tartibda ustunlarni joylashtirish
    input_data = input_data[model_features]

    # Bashorat
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    # Natijani chiqarish
    if prediction == 1:
        st.success(f"✅ Kredit **to‘lanadi** deb taxmin qilindi. Ehtimol: {prediction_proba:.2%}")
    else:
        st.error(f"❌ Kredit **to‘lanmaydi** deb taxmin qilindi. Ehtimol: {1 - prediction_proba:.2%}")
