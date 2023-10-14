import streamlit as st
import requests
import json
import time

def predict(prompt, img_height = 512, img_width = 512, num_inference_steps = 30, guidance_scale = 7.5):
    out = requests.post("http://localhost:5000/predict/",
                        data =json.dumps({"prompt":[prompt],
                                        "num_inference_steps":num_inference_steps,
                                        "guidance_scale":guidance_scale}
                                        ))
    return out.content

st.set_page_config(layout="wide")

st.title('Play with Stable-Diffusion with AIT+CK on AMD Navi31 XTX')

with st.form(key='new'):
    prompt = st.text_area(label='Enter prompt')
    cols = st.columns(2)
    num_inference_steps = cols[0].slider(
            "Number of Inference Steps (Image Generation Quality)",
            min_value=10,
            max_value=100,
            value=30,
            help="The more steps, the clearer and nicer the image will be"
        )
    guidance_scale = cols[1].slider(
            "Guidance Scale (Image Generation Precision)",
            min_value=2.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            help="The higher scale, the more closely the image follow the prompt"
        )
    submitted = st.form_submit_button("Generate")

    if submitted:
        st.image(
            predict(
                prompt,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
            ),
            caption='result'
        )
