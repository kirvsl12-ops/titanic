import streamlit as st
import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
model.load_state_dict(torch.load("model_weights_bath32_lr0.005_agegroup.pth"))
model.eval()
model.to(DEVICE)
st.title("Titanic Survival Prediction 🚢")
st.write("Введите данные пассажира")

def age_group(age):
    if age <= 3:
        return 0
    elif age <= 12:
        return 1
    elif age <= 18:
        return 2
    elif age <= 60:
        return 3
    else:
        return 4

ps = st.number_input("Место в титанике", value=0)
pclass = st.selectbox("Какой класс", [1, 2, 3])
se = st.selectbox("Пол", ["Мужской", "Женский"])
s = 0 if se == "Мужской" else 1
age = st.number_input("Сколько лет", value=0)
sibsp = st.number_input("Количество братьев, сестёр, мужей или жён, которые путешествовали вместе с вами", value=1)
parch = st.number_input("Количество родителей или детей, которых пассажир взял с собой на корабль", value=0)
ticket = st.number_input("Номер билета", value=0)
price = st.number_input("Цена билета", value=0)
em = st.selectbox("Порт посадки на корабль", ["C", "Q", "S"])
agegroup = age_group(age)
if em == "S":
    e = 0
elif em == "C":
    e = 1
else:
    e = 2 

inputs = [ps,pclass,age,s,sibsp,parch,ticket,price,e,agegroup]


if st.button("Predict"):
    x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    st.write(f"### Вероятность выжить: {prob * 100: .3f}%")
    if prob >= 0.3082:
        st.success("Пассажир вероятнее всего ВЫЖИВЕТ 🟢")
    else:
        st.error("Пассажир вероятнее всего НЕ ВЫЖИВЕТ 🔴")