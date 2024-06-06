import tkinter as tk
from tkinter import filedialog
from tkinter import *

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# "https://github.com/opencv/opencv/tree/master/data/haarcascades"

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.reas()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer = 'adam' , loss='categorical_crossentropy', metrics =['accuracy'])
    return model

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')
label1 = Label1(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image= Label(top)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel('model_a1.json','model_weights.h5')
EMOTIONS_LIST = ["Angry","Fear","happy","Neutral","sad","Surprise"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image= cv2.cvtcolor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.datectMultiScale(gray_image,1,3,5)
    try:
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,48,48)
            pred = EMOTIONS_LIST(np.argmax(model.predict(roi[np.newaxis,i,np.newaxis])))
        print("Predicted image is " + pred)
        label1.configure(foreground="#011638", text = pred)
    except:
        label1.configure(foreground="#011638", text ="unable to detect")
        
def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command = lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfile()
        uploaded = image.open(file_path)
        uploaded.thumbnail1(((top.winfo_width()/2.3),(top.winfo_height()/2.3)))
        im = imageTK.PhotoImage(uploaded)

        sign_image.configure(image = im)
        sign_image.image = im
        label1.configure(text= '')
        show_Detect_button(file_path)
    except:
        pass
upload = Button(top,text = "upload Image", command =upload_image, padx=10,pady=5)
upload.configure(background="#364156", foreground='white',font=('airal',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand='True')
label1.pack(side='botom',expand='True')
heading=label(top,text="Emotion Detector",pady=20, font =('airal',25,'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()