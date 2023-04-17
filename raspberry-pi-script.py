from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import experimental
from keras.callbacks import ModelCheckpoint, EarlyStopping
from roboflow import Roboflow
import matplotlib.pyplot as plt
import cv2
import time
import tkinter as tk
import os
from PIL import Image, ImageGrab
import turtle

from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import smtplib
from pathlib import Path

print("Imports Successful")


#Function that crops generated maps, then autonomously emails map to 
def dump_gui():
    print('...dumping gui window to png')

    x0 = root.winfo_rootx()+25
    y0 = root.winfo_rooty()+5
    x1 = x0 + (root.winfo_width()*1.5)
    y1 = y0 + (root.winfo_height()*1.5)
    ImageGrab.grab().crop((x0, y0, x1, y1)).save("turtle_img.png")

    email = "blightmanagementrow1@gmail.com"

    message = MIMEMultipart()
    message["from"] = email
    message["to"] = "ashank.shah@gmail.com"
    message["subject"] = "Row 1 Blight Update"
    message.attach(MIMEImage(Path("turtle_img.png").read_bytes()))

    print("Message Built")

    with smtplib.SMTP(host = "smtp.gmail.com", port = 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(user = email, password = "kwoowdrznwalkiwg")
        smtp.send_message(message)
        print("Sent!")



print("Dump Function Loaded")


#Initialize model architecture and load in weights
resize_and_rescale = Sequential([
  experimental.preprocessing.Resizing(256, 256),
  experimental.preprocessing.Rescaling(1./255),
])

data_augmentation = Sequential([
  experimental.preprocessing.RandomFlip("horizontal_and_vertical",input_shape = (32,256,256,3)),
  experimental.preprocessing.RandomRotation(0.1),
  experimental.preprocessing.RandomZoom(0.1),
  experimental.preprocessing.RandomTranslation(height_factor = 0.1, width_factor = 0.1, input_shape = (32,256,256,3))
])

input_shape = (32, 256, 256, 3)

classifier = Sequential([
    resize_and_rescale,
    data_augmentation,
    Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64,  kernel_size = (3,3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64,  kernel_size = (3,3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax'),
])

classifier.build(input_shape = input_shape)
classifier.load_weights("weights.h5")
print("Classifier Loaded")


#Load Object Detection Model API
rf = Roboflow(api_key="4w1xOJY7zRJkHhz0TVki")
project = rf.workspace().project("leaf-detection-1ssy5")
model = project.version(8).model

print("Object Detection Model Loaded")

#Function that isolates individual leaves in an image, then passes each leaf through the single leaf classifier
def predictplant(image):
    leaf_detections = model.predict(image, confidence=60, overlap=30).json()

    bounding_boxes = []

    for i in range(len(leaf_detections["predictions"])):
        current = leaf_detections["predictions"][i]
        bounding_boxes.append([current["x"], current["y"], current["width"], current["height"]])

    imarr = np.asarray(image)

    leafs = []
    for bounding_box in bounding_boxes:
        Xrange = int(bounding_box[1]-(bounding_box[3]/2))
        Xrange_end = int(bounding_box[1]+(bounding_box[3]/2))
        yrange = int(bounding_box[0]-(bounding_box[2]/2))
        yrange_end = int(bounding_box[0]+(bounding_box[2]/2))

        leafs.append(imarr[Xrange:Xrange_end,yrange:yrange_end])

    leaf_arr = []
    for leaf in leafs:
        current_im = Image.fromarray(leaf).resize((256,256))
        leaf_arr.append(np.asarray(current_im))

    preds = []
    for i in range(len(leaf_arr)):
        arr = np.array([np.asarray(leaf_arr[i])])
        preds.append(np.argmax(classifier.predict(arr)))

    if len(preds) == 0:
        return "No leafs"
    elif preds.count(1)>0:
        return "Diseased", len(leafs)
    else:
        return "Healthy", len(leafs)

print("Plant Function Defined")


#Map Generation
root = tk.Tk()
canvas = tk.Canvas(root, width=400, height=570)
canvas.pack()

robot = turtle.RawTurtle(canvas)
robot.shape("circle")
robot.hideturtle()
robot.shapesize(0.5,0.5,0.5)
robot.speed(0)
robot.left(90)
robot.penup()
robot.goto(0,-280)
robot.speed(1)
robot.showturtle()

draw = turtle.RawTurtle(canvas)
draw.speed(0)
draw.color("black")
draw.hideturtle()
draw.penup()


draw.goto(-75,-300)
draw.pendown()
draw.left(90)
draw.forward(600)
draw.penup()

draw.goto(75,-300)
draw.pendown()
draw.forward(600)
draw.penup()

draw.left(90)


for i in range(10):
    draw.penup()
    draw.goto(-75, 57*i-285)
    draw.pendown()
    draw.forward(30)


draw.penup()

draw.color("red")

#Start video input
vid = cv2.VideoCapture(0)

#When a diseased leaf is found in a given frame, a red dot is drawn
for i in range(14):
    ret, frame = vid.read()
    #cv2.imshow('frame', frame)
    classification = predictplant(frame)
    print(classification)
    time.sleep(2)
    robot.forward(37.5)
    if classification == "Diseased":
        draw.goto(robot.xcor(), robot.ycor())
        draw.pendown()
        draw.begin_fill()
        draw.circle(2)
        draw.end_fill()
        draw.penup()
    time.sleep(7)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

dump_gui()

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()