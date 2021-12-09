import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import imageio
import base64
import matplotlib.pyplot as plt
from PIL import Image
from clip_beam_grounding_v2 import CLIPBeamGrounding
from matplotlib.patches import Rectangle

def visualize_resions(img_path, box_list, clipscore_list):
    image_file_path=img_path
    img = Image.open(image_file_path)
    for i, box in enumerate(box_list):
        plt.imshow(img)
        boxax=plt.gca()
        w=box[2]-box[0]
        h=box[3]-box[1]
        boxax.add_patch(Rectangle((box[0],box[1]),
                        w,
                        h,
                        fill=False,
                        edgecolor='red',
                        linewidth=3))
        boxax.text(box[0], box[1], 'step '+str(i)+'\nclip score '+str(clipscore_list[i]), style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
        plt.savefig(str(i)+'.jpg')
        plt.show()   
        boxax.clear() 


if __name__ == '__main__':
    st.title("Demo")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model =  "ViT-B/32"
    cg = CLIPBeamGrounding(clip_model, device, alpha=0.2)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width='auto')

    query=st.text_input('please input things you want to find in your picture. such as \"people\" or something')
    steps=st.number_input('please input a number means the maxmum steps during the search', 
                        min_value=1, max_value=100, value=20, step=1)
    beam=st.number_input('please input a number means the beam size during the search', 
                        min_value=1, max_value=len(cg.choices), value=3, step=1)
    patience=st.number_input('please input a number means the patience during the search', 
                        min_value=0, max_value=5, value=3, step=1)

    if st.button('Start Search'):
        box_list, clip_scores = cg.search(img_path=uploaded_file, text_query=query, beam_size=beam, 
                                          patience=patience, max_steps=steps)

        visualize_resions(uploaded_file, box_list, clip_scores)
        img=list()
        cap=list()
        frames=list()
        files = os.listdir('./')
        picfiles=[]
        for f in files:
            if f[-4:]=='.jpg':
                picfiles.append(f)
        picfiles.sort(key=lambda x:int(x[:-4]))
        # print(picfiles)

        for image_name in picfiles:
            im = imageio.imread(image_name)
            frames.append(im)
        if frames:
            imageio.mimsave('search.gif',frames,'GIF',duration=1)
        else:
            print('no support images!')


        #https://github.com/streamlit/streamlit/issues/1566
        file_ = open("search.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )



        for i in range(len(box_list)):
            img.append(Image.open(str(i)+'.jpg'))
            cap.append(str(i))
            os.system('rm '+str(i)+'.jpg')
        st.image(img, caption=cap)
