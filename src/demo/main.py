import streamlit as st
import pandas as pd
import numpy as np

from ..inference.e2e import e2e

st.title("KoUniPunc Demo")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    st.audio(bytes_data, format="audio/ogg")

    if st.button("Convert"):
        con = st.container()
        con.caption("Result")

        e2e(bytes_data)

        con.write("hello~")
