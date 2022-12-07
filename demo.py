import streamlit as st

from e2e_model import e2e_inference

if __name__ == '__main__':    
    st.title("KoUniPunc Demo")
    
    uploaded_file = st.file_uploader("Choose a file!")
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Convert"):
            con = st.container()
            con.caption("Punctuation Result")

            with st.spinner("Punctuation in progress..."):
                result = e2e_inference(uploaded_file)
            con.write(result)
