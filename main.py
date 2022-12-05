import streamlit as st
from audio_recorder_streamlit import audio_recorder

from e2e_model import e2e_inference

st.title("KoUniPunc Demo")

audio_bytes = audio_recorder(text="Click to Record")
uploaded_file = st.file_uploader("Or choose a file!")


def convert_audio(audio):
    st.audio(audio, format="audio/wav")

    if st.button("Convert"):
        con = st.container()
        con.caption("Punctuation Result")

        with st.spinner("Punctuation in progress..."):
            result = e2e_inference(audio)
        con.write(result)


if uploaded_file is not None:
    convert_audio(uploaded_file)

elif audio_bytes is not None:
    convert_audio(audio_bytes)
