import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

class Frontend:
    def __init__(self):
        self.load_image=tf.keras.models.load_model('model_cifar10_v2.h5')
        self.class_names=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
    
    def image(self,image):
        img=image.resize((32,32))
        img=np.array(img)/255.0
        img=np.expand_dims(img,axis=0)
        return  img
    
    def predicts(self,image):
        preds=self.load_image.predict(image)[0]
        top_index=np.argmax(preds)
        return self.class_names[top_index],preds
    
    def barchart(self,preds):
        chart=pd.DataFrame({"Types":self.class_names,'Result':preds})
        st.bar_chart(chart.set_index('Types'))

    def upload(self):
        st.title("Image Recognition")
        st.write('Upload image to recognize the image')
        upload_image=st.file_uploader('upload image',type=['jpg','png','jpeg'])
        if upload_image:
            image=Image.open(upload_image)
            st.image(image,width=300)
            img_array=self.image(image)
            prediction,preds=self.predicts(img_array)
            self.barchart(preds)



if __name__=='__main__':
    f=Frontend()
    f.upload()