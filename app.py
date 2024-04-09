from backend import *
import io

import streamlit as st

st.set_page_config(
    page_title="ROI",
    page_icon="🤖",
    )
st.subheader("🤖 ROI approximation from attention weights")

with st.spinner("Loading..."):
    model, processor, tokenizer = load_model()

def execute():
    image, caption = extract_roi(st.session_state['image'], processor, tokenizer, model)
    image = square_crop(image, 100)

    st.session_state['output'] = image
    st.session_state['caption'] = caption

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = io.BytesIO(uploaded_file.read())
    print("loaded image")
    st.session_state['image'] = Image.open(bytes_data)

if 'image' in st.session_state:
    with st.expander("Your Image", expanded=True):
        st.write("uploaded image")
        st.image(st.session_state['image'])

    bt = st.button("Run query", on_click=execute)


if 'output' in st.session_state:
    with st.expander("Output", expanded=True):
        st.write("Generated Caption")
        st.write(st.session_state['caption'])

        st.image(st.session_state['output'])
        