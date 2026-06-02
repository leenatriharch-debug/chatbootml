import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("customersupport.csv")
print(df.head(20))

X = df["question"]
y = df["answer"]

# Vectorization + model training
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# UI
st.title("💬 Customer Support Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask your question:")

if st.button("Send"):
    if user_input:

        user_vec = vectorizer.transform([user_input])
        scores = model.predict_proba(user_vec)[0]
        max_score = max(scores)

        if max_score < 0.3:
            response = "Sorry, I don't understand. Please rephrase."
        else:
            response = model.predict(user_vec)[0]

        st.session_state.chat.append(("You", user_input))
        st.session_state.chat.append(("Bot", response))

# Chat history show
for role, msg in st.session_state.chat:
    if role == "You":
        st.write(f" {msg}")
    else:
        st.write(f"{msg}")
        #C:\Users\Dell Latitude\Desktop\project of ml\chat-boot\CustomerSumpportBot"
       # -m streamlit run app.py

       