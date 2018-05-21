import numpy as np
import math
import cv2 
import matplotlib.pyplot as plt
import scipy.spatial.distance as scipy
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy.ndimage import  median_filter
from matplotlib import gridspec
import multiprocessing
from joblib import Parallel, delayed
from skimage.feature import greycomatrix, greycoprops
import sys

class Image():
    def __init__(self,img, drawArea):
        self.oldImg = img * 1
        self.img = img
        self.drawArea = drawArea    
    
    def setImage(self,img):
        self.oldImg = img * 1
        self.img = img

    def resetImg(self):
        self.img = self.oldImg * 1
        self.update()
        
    def update(self):
        self.drawArea.clear()
        self.drawArea.axis("off")
        self.drawArea.imshow(self.img, cmap='gray', vmin = 0, vmax = 255)

    def median_filter(self):
        self.img = median_filter(self.img, size = (3,3))
        return self.img

    def box_filter(self):
        self.img = cv2.boxFilter(self.img, ddepth = -1, ksize = (3,3))
        return self.img

    def calculateCoMatrix(self):
        self.comatrix = greycomatrix(self.img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=256)
        self.totalCoMatrix = np.zeros((4,1))
        for p in range(0,4):
            for i in range(0,256):
                for j in range(0,256):
                    self.totalCoMatrix[p] +=  self.comatrix[i][j][0][p]

        
    def calculateCoHomogenity(self):
        homogenity = greycoprops(self.comatrix, 'homogeneity')
        self.h0 = homogenity[0][0]
        self.h45 = homogenity[0][1]
        self.h90 = homogenity[0][2]
        self.h135 = homogenity[0][3]

    def calculateUniformity(self):
        n = self.getTotalCO()
        self.uniformity = np.zeros((4,1))
        
        for p in range(0,4):
            for i in range(0,256):
                for j in range(0,256):
                    self.uniformity[p] += math.pow( (self.comatrix[i][j][0][p]/n[p]),2)

        
    def calculateHomogenity(self):
        n = self.getTotalCO()
        self.homogenity = np.zeros((4,1))
        
        for p in range(0,4):
            for i in range(0,256):
                for j in range(0,256):
                    self.homogenity[p] += (self.comatrix[i][j][0][p]/n[p]) /  (1+abs(i-j))

    def getUniformity(self):
        return self.uniformity

    def getHomogenity(self):
        return self.homogenity

    def getCoHomogenity(self,angle):
        if angle == 0:
            return self.h0
        elif  angle == 45:
            return self.h45
        elif  angle == 90:
            return self.h90
        elif  angle == 135:
            return self.h135
        else:
            print 'Valor invalido'
    
    def getCoMatrix(self):
        return self.comatrix

    def getTotalCO(self):
        return self.totalCoMatrix

def update(val):
    global lImg, slPic
    i = int(round(slPic.val))
    lImg[i].update()
    changeChart('sajd')
    plt.show()

def calculateImg():
    global myImg, lMedian, lBox, lRM, LRB, a0, lImg
    myImg.resetImg()
    myImg.calculateCoMatrix()
    myImg.calculateCoHomogenity()
    myImg.calculateUniformity()
    cur = myImg.img
    # Calcula median 
    for i in range (0,30):
        if i != 0:
            myImg.median_filter()

        img = Image(myImg.img,a0)
        r = Image(cur - img.img, a0)

        img.calculateCoMatrix()
        img.calculateCoHomogenity()
        img.calculateUniformity()
        img.calculateHomogenity()
        lMedian.append(img)

        r.calculateCoMatrix()
        r.calculateHomogenity()
        r.calculateUniformity()
        r.calculateCoHomogenity()
        lRM.append(r)

    myImg.resetImg()

    # Calculate Box Filter
    for i in range (0,30):
        if i != 0:
            myImg.box_filter()

        img = Image(myImg.img,a0)
        r = Image(cur - img.img, a0)

        img.calculateCoMatrix()
        img.calculateCoHomogenity()
        img.calculateUniformity()
        img.calculateHomogenity()
        lBox.append(img)

        r.calculateCoMatrix()
        r.calculateUniformity()
        r.calculateHomogenity()
        r.calculateCoHomogenity()
        LRB.append(r)
    
    myImg.resetImg()

def show(label):
    global lMedian,lBox, myImg, slPic, lImg
    
    if label == 'Box':
        lImg = lBox
    else:
        lImg = lMedian

    update(slPic.val)

def reset(event):
    global slPic
    slPic.reset()

def constructDatas():
    global slPic, myImg, lBox, lMedian, DMedian, DBox, lRM, LRB, dRB, dRM
    total = myImg.getTotalCO()

    # Data [0] x [1] .. [8] y - Homogenity, Uniformity
    ax = range(0,30)
    DBox.append(ax)
    DMedian.append(ax)
    dRB.append(ax)
    dRM.append(ax)

    DBox.append([])
    DBox.append([])
    DBox.append([])
    DBox.append([])
    DBox.append([])
    DBox.append([])
    DBox.append([])
    DBox.append([])

    DMedian.append([])
    DMedian.append([])
    DMedian.append([])
    DMedian.append([])
    DMedian.append([])
    DMedian.append([])
    DMedian.append([])
    DMedian.append([])

    dRB.append([])
    dRB.append([])
    dRB.append([])
    dRB.append([])
    dRB.append([])
    dRB.append([])
    dRB.append([])
    dRB.append([])

    dRM.append([])
    dRM.append([])
    dRM.append([])
    dRM.append([])
    dRM.append([])
    dRM.append([])
    dRM.append([])
    dRM.append([])
    
    for i in lRM:
        dRB[1].append(i.getHomogenity()[0])
        dRB[2].append(i.getHomogenity()[1])
        dRB[3].append(i.getHomogenity()[2])
        dRB[4].append(i.getHomogenity()[3])

        dRB[5].append(i.getUniformity()[0]/total[0])
        dRB[6].append(i.getUniformity()[1]/total[1])
        dRB[7].append(i.getUniformity()[2]/total[2])
        dRB[8].append(i.getUniformity()[3]/total[3])
    
    
    for i in LRB:
        dRM[1].append(i.getHomogenity()[0])
        dRM[2].append(i.getHomogenity()[1])
        dRM[3].append(i.getHomogenity()[2])
        dRM[4].append(i.getHomogenity()[3])

        dRM[5].append(i.getUniformity()[0]/total[0])
        dRM[6].append(i.getUniformity()[1]/total[1])
        dRM[7].append(i.getUniformity()[2]/total[2])
        dRM[8].append(i.getUniformity()[3]/total[3])
    
    
    for i in lBox:
        DBox[1].append(i.getHomogenity()[0])
        DBox[2].append(i.getHomogenity()[1])
        DBox[3].append(i.getHomogenity()[2])
        DBox[4].append(i.getHomogenity()[3])

        DBox[5].append(i.getUniformity()[0]/total[0])
        DBox[6].append(i.getUniformity()[1]/total[1])
        DBox[7].append(i.getUniformity()[2]/total[2])
        DBox[8].append(i.getUniformity()[3]/total[3])
    
    for i in lMedian:
        DMedian[1].append(i.getHomogenity()[0])
        DMedian[2].append(i.getHomogenity()[1])
        DMedian[3].append(i.getHomogenity()[2])
        DMedian[4].append(i.getHomogenity()[3])

        DMedian[5].append(i.getUniformity()[0]/total[0])
        DMedian[6].append(i.getUniformity()[1]/total[1])
        DMedian[7].append(i.getUniformity()[2]/total[2])
        DMedian[8].append(i.getUniformity()[3]/total[3])


        

def changeChart(label):
    global radio,b0,DMedian,DBox, dRM, dRB, radio, radio2, radio3, radio4, myImg
    tipo = radio.value_selected
    angle = radio2.value_selected
    variable = radio3.value_selected
    RS = radio4.value_selected
    total = myImg.getTotalCO
    
    dataY = []
    if RS == 'R':
        if tipo == 'Box':
            if variable == 'Homogenity':
                if angle == '0':
                    dataY = dRB[1]
                elif angle == '45':
                    dataY = dRB[2]
                elif angle == '90':
                    dataY = dRB[3]
                else:
                    dataY = dRB[4]
            else:
                if angle == '0':
                    dataY = dRB[5]
                elif angle == '45':
                    dataY = dRB[6]
                elif angle == '90':
                    dataY = dRB[7]
                else:
                    dataY = dRB[8]
        else:
            if variable == 'Homogenity':
                if angle == '0':
                    dataY = dRM[1]
                elif angle == '45':
                    dataY = dRM[2]
                elif angle == '90':
                    dataY = dRM[3]
                else:
                    dataY = dRM[4]
            else:
                if angle == '0':
                    dataY = dRM[5]
                elif angle == '45':
                    dataY = dRM[6]
                elif angle == '90':
                    dataY = dRM[7]
                else:
                    dataY = dRM[8]
    else:
        if tipo == 'Box':
            if variable == 'Homogenity':
                if angle == '0':
                    dataY = DBox[1]
                elif angle == '45':
                    dataY = DBox[2]
                elif angle == '90':
                    dataY = DBox[3]
                else:
                    dataY = DBox[4]
            else:
                if angle == '0':
                    dataY = DBox[5]
                elif angle == '45':
                    dataY = dRB[6]
                elif angle == '90':
                    dataY = DBox[7]
                else:
                    dataY = DBox[8]
        else:
            if variable == 'Homogenity':
                if angle == '0':
                    dataY = DMedian[1]
                elif angle == '45':
                    dataY = DMedian[2]
                elif angle == '90':
                    dataY = DMedian[3]
                else:
                    dataY = DMedian[4]
            else:
                if angle == '0':
                   dataY = DMedian[5]
                elif angle == '45':
                    dataY = DMedian[6]
                elif angle == '90':
                    dataY = DMedian[7]
                else:
                    dataY = DMedian[8]
    
    b0.clear()
    b0.plot(DMedian[0],dataY)
    plt.show

def main():
    global slPic, a0,b0, lImg, lMedian, lBox, myImg,  lRM, LRB, radio, radio2, radio3, radio4, dRM, dRB, DMedian, DBox
    sys.setrecursionlimit(100000)
    
    lRM = [] 
    LRB = []
    lBox = []
    DBox = []
    lMedian = []
    DMedian = []
    dRM = []
    dRB = []
    
    img = cv2.imread('pig.jpg')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    plt.figure()
    plt.axis("off")
    plt.get_current_fig_manager().window.wm_geometry("-1200-500")
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
    plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update( hspace=1.5, wspace=1.5)
    a0 = plt.subplot(gs[:-1,: -2])
    a1 = plt.subplot(gs[9,:7])
    a2 = plt.subplot(gs[:2, 8:10])
    a3 = plt.subplot(gs[9, 8:10])
    plt.get_current_fig_manager().window.wm_geometry("-550-500")

    plt.figure()
    gs = gridspec.GridSpec(10,12)
    gs.update( hspace=1.5, wspace=1.5)
    b0 = plt.subplot(gs[:,: -3])
    b1 = plt.subplot(gs[:4, 9:12])
    b2 = plt.subplot(gs[5:7, 9:12])
    b3 = plt.subplot(gs[8:10, 9:12])
    
    plt.get_current_fig_manager().window.wm_geometry("-0-500")

    myImg = Image(gray_img,a0)
    myImg.update()
    
    ret,thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY)
    a0.imshow(thresh, cmap = 'gray',  vmin = 0, vmax = 255)
    
    slPic = Slider(a1,'Picture', valmin=0,valmax=29, valinit=0, valfmt= "%3.0f")
    
    radio = RadioButtons(a2,('Box', 'Median'),active=1)
    radio2 = RadioButtons(b1,('0', '45', '90','135' ))
    radio3 = RadioButtons(b2,('Homogenity', 'Uniformity'))
    radio4 = RadioButtons(b3,('R', 'S'))
    button = Button(a3,'Reset', color='gray', hovercolor='0.8')
    
    button.on_clicked(reset)
    slPic.on_changed(update)
    radio.on_clicked(show)
    radio2.on_clicked(changeChart)
    radio3.on_clicked(changeChart)
    radio4.on_clicked(changeChart)
    calculateImg()
    constructDatas()
    show('Box Filter')
    update(0)
    plt.show()

if __name__ == "__main__":
    main()