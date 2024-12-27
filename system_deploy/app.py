#import cv2
import streamlit as st
from PIL import Image
from clf import predict
import matplotlib.pyplot as plt
import numpy as np
import time
# st.set_option('deprecation.showfileUploaderEncoding', False)

# st.title("糖尿病视网膜病变系统")
st.markdown("<h1 style='text-align: center; color: grey;'>遥感图像道路提取系统</h1>", unsafe_allow_html=True)
st.write("")
st.write("")
# option = st.selectbox(
#      'Choose the model you want to use?',
#      ('resnet50', 'resnet101', 'densenet121','shufflenet_v2_x0_5','mobilenet_v2'))
# ""
# option2 = st.selectbox(
#      'you can select some image',
#      ('image_dog', 'image_snake'))

# st.write("请上传一张图片：")
file_up = st.file_uploader("请上传一张图片")

# image = Image.open("image/google.jpg")
# st.image(image, caption='Uploaded Image.', use_column_width=True)
if file_up is not None:
    col1,col2 = st.columns(2)
    
    
    # image = Image.open("image/10378780_15.tiff")
    image = Image.open(file_up)
    # st.write(file_up)
    with col1:
        st.image(image, use_container_width=True)
        st.write("Uploded Image.")
    # st.write("Just a second...")

    mask = predict(image)
    # labels, fps = predict(file_up, option)
    st.success('成功预测，请等待结果显示')

    # print out the top 5 prediction labels with scores
    # st.success('successful prediction')


    # st.pyplot(mask,cmap='gray', vmin=0, vmax=1)

    # st.image(std, caption='Uncertainty Map', use_column_width=True,clamp=True)
    
    # st.write("分割结果如下：")
    # st.image(mask, caption='Uploaded Image.', use_container_width=True)
    with col2:
        fig, ax = plt.subplots()
        im = ax.imshow(mask, cmap='binary', vmin=0, vmax=1)
        ax.axis('off')  # 隐藏坐标轴
        st.pyplot(fig,use_container_width=True)
        st.write("Result.")
        


    # st.write("")
    # st.metric("","FPS:   "+str(fps)

else:
    # image = Image.open("image/google.jpg")
        # cv2.imread("image/Image_01L.jpg", cv2.IMREAD_COLOR)
    # image = image[:,:,-1]
    # st.write("没有上传图片文件，请上传")
    # if option2 =="image_dog":
    #     image=Image.open("image/dog.jpg")
    #     file_up="image/dog.jpg"
    # else:
    #     image=Image.open("image/snake.jpg")
    #     file_up="image/snake.jpg"
    # st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.warning("上传为空，请上传图像!")
    # st.write()

#     fps, prior, std, mask = predict(file_up)
  
#     st.write("")
#     st.metric("", "FPS:   " + str(fps))