import streamlit as st
import requests


st.title("KoUniPunc Demo")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    st.audio(bytes_data, format="audio/ogg")

    if st.button("Convert"):
        con = st.container()
        con.caption("Result")

        result: str = requests.post("http://server:8080", files={"file": uploaded_file})
        con.write(result)
