#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
res = requests.get('https://www.google.com/search?ei=d3IqXMGIG8y4sQWzuo2IDw&yv=3&q=%E9%88%95%E6%89%BF%E6%BE%A4&tbm=isch&vet=10ahUKEwiBoPyI68rfAhVMXKwKHTNdA_EQuT0IOSgB.d3IqXMGIG8y4sQWzuo2IDw.i&ved=0ahUKEwiBoPyI68rfAhVMXKwKHTNdA_EQuT0IOSgB&ijn=1&start=100&asearch=ichunk&async=_id:rg_s,_pms:s,_fmt:pc')


# In[2]:


from bs4 import BeautifulSoup
soup = BeautifulSoup(res.text, 'lxml')


# In[3]:


for ele in soup.select('img'):
    print(ele.get('src'))


# In[4]:


with open('1.jpg', 'wb') as f:
    res = requests.get('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRenJ2NhFqaqhsaDLFa06pNtqKgY_A_WwRmgH4QZrYpmsEPcAsFNQ')
    f.write(res.content)


# In[5]:


dataurl = 'https://www.google.com/search?ei=d3IqXMGIG8y4sQWzuo2IDw&yv=3&q={}&tbm=isch&vet=10ahUKEwiBoPyI68rfAhVMXKwKHTNdA_EQuT0IOSgB.d3IqXMGIG8y4sQWzuo2IDw.i&ved=0ahUKEwiBoPyI68rfAhVMXKwKHTNdA_EQuT0IOSgB&ijn=1&start={}&asearch=ichunk&async=_id:rg_s,_pms:s,_fmt:pc'
def getIdolImg(keyword, dstpath):
    for i in range(1):
        res = requests.get(dataurl.format(keyword,i*100))
        soup = BeautifulSoup(res.text, 'lxml')
        for ele in soup.select('img'):
            imageurl = ele.get('src')
            fname = imageurl.split('tbn:')[1]
            with open(dstpath + fname + '.jpg', 'wb') as f:
                res2 = requests.get(imageurl)
                f.write(res2.content)


# In[6]:


import os
os.mkdir('idol1/')


# In[7]:


getIdolImg('鈕承澤', 'idol1/')


# In[8]:


import os
os.listdir('idol1')[0:8]


# In[20]:


from PIL import Image
img = Image.open('idol1/ANd9GcR4eOzDASTIZWnlYoisjJNrJlRULMaETnUaPGYCRAaWOBLbKAPe.jpg')
img


# In[21]:


import cv2 as cv
imgary = cv.imread('idol1/ANd9GcR4eOzDASTIZWnlYoisjJNrJlRULMaETnUaPGYCRAaWOBLbKAPe.jpg')


# In[22]:


imgary


# In[23]:


imgary.shape


# In[50]:


face_cascade = cv.CascadeClassifier('/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')


# In[52]:


faces = face_cascade.detectMultiScale(imgary, 1.3, 5)


# In[53]:


faces


# In[54]:


x,y,w,h = faces[0]


# In[55]:


crpim = img.crop((x,y, x + w, y + h)).resize((64,64))
crpim


# In[56]:


import os
os.mkdir('idol1_face/')


# In[60]:


import os
srcpath = 'idol1/' 
dstpath = 'idol1_face/'
face_cascade = cv.CascadeClassifier('/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
for fname in os.listdir(srcpath):
    img = Image.open(srcpath + fname)
    imgary = cv.imread(srcpath + fname)
    faces = face_cascade.detectMultiScale(imgary, 1.3, 5)
    if len(faces) == 1:
        x,y,w,h = faces[0]
        crpim = img.crop((x,y, x + w, y + h)).resize((64,64))
        crpim.save(dstpath + fname)
    #print(srcpath + fname)


# In[61]:


os.listdir(dstpath)[0:5]


# In[62]:


Image.open('idol1_face/ANd9GcQ4RmDmDwEVskMlCg53N5cREt2Mw2PwPi1mnimBweZMDc-qjBEq.jpg')


# In[64]:


import os
import requests
from bs4 import BeautifulSoup
dataurl = 'https://www.google.com/search?ei=d3IqXMGIG8y4sQWzuo2IDw&yv=3&q={}&tbm=isch&vet=10ahUKEwiBoPyI68rfAhVMXKwKHTNdA_EQuT0IOSgB.d3IqXMGIG8y4sQWzuo2IDw.i&ved=0ahUKEwiBoPyI68rfAhVMXKwKHTNdA_EQuT0IOSgB&ijn=1&start={}&asearch=ichunk&async=_id:rg_s,_pms:s,_fmt:pc'

def getIdolImg(keyword, dstpath):
    for i in range(6):
        res = requests.get(dataurl.format(keyword, i * 100))
        soup = BeautifulSoup(res.text, 'lxml')
        for ele in soup.select('img'):
            imgurl = ele.get('src')
            fname  = imgurl.split('tbn:')[1]
            with open(dstpath + fname + '.jpg', 'wb') as f:
                res2 = requests.get(imgurl)
                f.write(res2.content)


# In[65]:


if not os.path.exists('NiuChenZer/'):
    os.mkdir('NiuChenZer/')
getIdolImg('鈕承澤', 'NiuChenZer/')


# In[66]:


if not os.path.exists('ChuChungHeng'):
    os.mkdir('ChuChungHeng/')
getIdolImg('屈中恆', 'ChuChungHeng/')


# In[67]:


if not os.path.exists('SungShaoChing'):
    os.mkdir('SungShaoChing/')
getIdolImg('宋少卿', 'SungShaoChing/')


# In[68]:


import cv2 as cv
from PIL import Image

def extractFace(srcpath, dstpath):
    if not os.path.exists(srcpath):
        os.mkdir(srcpath)
    if not os.path.exists(dstpath):
        os.mkdir(dstpath)
    face_cascade = cv.CascadeClassifier('/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    for fname in os.listdir(srcpath):
        img = Image.open(srcpath + fname)
        imgary = cv.imread(srcpath + fname)
        faces = face_cascade.detectMultiScale(imgary, 1.3, 5)
        if len(faces) == 1:
            x,y,w,h = faces[0]
            crpim = img.crop((x,y, x + w, y + h)).resize((64,64))
            crpim.save(dstpath + fname)


# In[69]:


srcpath = 'NiuChenZer/' 
dstpath = 'NiuChenZerFace/'
extractFace(srcpath, dstpath)


# In[70]:


srcpath = 'ChuChungHeng/' 
dstpath = 'ChuChungHengFace/'
extractFace(srcpath, dstpath)


# In[71]:


srcpath = 'SungShaoChing/' 
dstpath = 'SungShaoChingFace/'
extractFace(srcpath, dstpath)


# In[72]:


get_ipython().system(' pip3 install tensorflow')


# In[73]:


get_ipython().system(' pip3 install keras')


# In[74]:


from keras.models import Sequential 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64,
3), activation = 'relu'))

# Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Convolution
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

# Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Flattening
classifier.add(Flatten())

# Fully Connected
classifier.add(Dense(units = 128, activation = 'relu')) 
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

classifier.compile(optimizer = 'adam', 
                        loss ='categorical_crossentropy', 
                     metrics = ['accuracy'])


# In[75]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,     
                                   zoom_range = 0.2,      
                                   horizontal_flip = True 
                                  )


# In[76]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[79]:


training_set = train_datagen.flow_from_directory(
    'training/', target_size = (64, 64),
     batch_size = 10,
     class_mode = 'categorical')


# In[80]:


test_set = test_datagen.flow_from_directory(
    'testing/', target_size = (64, 64),
    batch_size = 10, 
    class_mode = 'categorical')


# In[81]:


history = classifier.fit_generator(training_set,
                         nb_epoch=10,
                         nb_val_samples=10,
                         steps_per_epoch = 30,
                         verbose = 1,
                         validation_data = test_set)


# In[83]:


from PIL import Image
im = Image.open('whoami.jpg')
im


# In[85]:


from PIL import Image
import cv2 as cv
face_cascade = cv.CascadeClassifier('/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

img = cv.imread('whoami.jpg')
faces = face_cascade.detectMultiScale(img, 1.2, 3)
faces


# In[86]:


transform_dic = {
    'ChuChungHengFace'  : 'Chu Chung-Heng',
    'NiuChenZerFace'    : 'Niu Chen-Zer',
    'SungShaoChingFace' : 'Sung Shao-Ching'
}
name_dic = {v:transform_dic.get(k) for k,v in training_set.class_indices.items()}
name_dic


# In[87]:


from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt 
font = cv.FONT_HERSHEY_PLAIN
for x,y,w,h in faces:
    box = (x, y, x+w, y+h)
    crpim = im.crop(box).resize((64,64))
    target_image = image.img_to_array(crpim)
    target_image = np.expand_dims(target_image, axis = 0)
    res = classifier.predict_classes(target_image)[0]
    cv.rectangle(img,(x,y),(x+w,y+h),(14,201,255),2)
    cv.putText(img,name_dic.get(res), (x + int(w/3)-70, y-10), font, 1.5, (14,201,255), 3)


# In[92]:


get_ipython().run_line_magic('pylab', 'inline')
plt.figure(figsize=(30,20))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

