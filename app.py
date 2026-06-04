import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from streamlit_local_storage import LocalStorage

localS = LocalStorage()

df = pd.read_csv("customersupport.csv")

X = df["question"]
y = df["answer"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

st.title("💬 Customer Support Chatbot")

# ---------- NAME ----------
name = localS.getItem("user_name")

if not name:

    user_name = st.text_input("Enter Your Name")

    # 👉 LIVE GREETING WHILE TYPING
    if user_name:
        st.write("👋 Welcome", user_name)

    if st.button("Save Name") and user_name:
        localS.setItem("user_name", user_name)
        st.rerun()

else:

    st.write("👋 Welcome back", name)

    if "chat" not in st.session_state:
        old_chat = localS.getItem("chat_history")

        if old_chat:
            st.session_state.chat = old_chat
        else:
            st.session_state.chat = [
                ("Bot", "Hello " + name + "! How can I help you?")
            ]

    question = st.text_input("Ask your question")

    if st.button("Send") and question:

        user_vec = vectorizer.transform([question])
        score = max(model.predict_proba(user_vec)[0])

        if score < 0.3:
            answer = "Sorry, I don't understand."
        else:
            answer = model.predict(user_vec)[0]

        st.session_state.chat.append(("You", question))
        st.session_state.chat.append(("Bot", answer))

        localS.setItem("chat_history", st.session_state.chat)
        st.rerun()

    for role, msg in st.session_state.chat:
        if role == "You":
            st.write("🧑 You:", msg)
        else:
            st.write("🤖 Bot:", msg)

    if st.button("Clear Chat"):
        st.session_state.chat = [
            ("Bot", "Hello " + name + "! How can I help you?")
        ]
        localS.setItem("chat_history", st.session_state.chat)
        st.rerun()