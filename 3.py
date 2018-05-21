import numpy as np
import math
import cv2 
import matplotlib.pyplot as plt
import scipy.spatial.distance as scipy
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib import gridspec
import multiprocessing
from joblib import Parallel, delayed
import sys


def sqdist(v):
    return  math.sqrt(sum(x**2 for x in v))

def findDist(o,plist):
    ox,oy = o
    for n in plist:
        nx,ny = n
        if abs(ox-nx) <= 1 and abs(oy-ny) <= 1:
            if abs(ox-nx) == 1:
                if oy == ny:
                    return 1
                elif abs(oy-ny) == 1:
                    return math.sqrt(2)
            elif abs(oy-ny) == 1:
                return 1
            elif o == n:
                return 1

class Component():
    def __init__(self,img):
        self.lpixels = []
        self.lborder = []
        self.diameter = 0
        self.area = 0
        self.perimeter = 0
        self.img = img

    def add(self, pixel):
        self.lpixels.append(pixel)

    def addB(self, pixel):
        self.lborder.append(pixel)

    def findCentroid(self):
        self.centroid = np.zeros((2))
        total = 0
        for i in self.lpixels:
            self.centroid[0] += self.img.getPixel(i[1],i[0]) * i[0]
            self.centroid[1] += self.img.getPixel(i[1],i[0]) * i[1]
            total += self.img.getPixel(i[1],i[0])
        
        
        self.centroid[0] = int(round(self.centroid[0]/total))
        self.centroid[1] = int(round(self.centroid[1]/total))

        self.teta = 0

        for i in self.lpixels:
            self.teta += 2 * self.img.getPixel(i[1],i[0]) * (i[0] - self.centroid[0]) * (i[1] - self.centroid[1])
        
        divisor1 = 0
        divisor2 = 0

        for i in self.lpixels:
            divisor1 +=  self.img.getPixel(i[1],i[0]) * pow((i[0] - self.centroid[0]),2)

        for i in self.lpixels:
            divisor2 +=  self.img.getPixel(i[1],i[0]) * pow((i[1] - self.centroid[1]),2)

        self.teta =(math.atan(self.teta/(divisor1-divisor2)))/2 
    def getCentroid(self):
        return self.centroid
    
    def getArea(self):
        return self.area
    def getPerimeter(self):
        return self.perimeter
    def getDiameter(self):
        return self.diameter
    
    def calulateArea(self):
        return len(self.lpixels)
    
    def calculatePerimeter(self):
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(findDist)(a,self.lborder) for a in self.lborder)
        
        return np.sum(results)

    def calculateDiameter(self):
        diameter = 0
        max = len(self.lborder)
        for i in range(0,max):
            for j in range(i,max):
                if i!=j:
                    temp = scipy.euclidean(self.lborder[i],self.lborder[j])
                    if temp > diameter:
                        diameter = temp
        return diameter

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
        
    def getShape(self):
        return self.img.shape
    
    def getPixel(self, i,j):
        return self.img[i][j]

    def removePixels(self,pixels):
        for p in pixels:
            x,y = p
            self.img[y][x] = 254

    def makePoint(self,x,y):
        rows,cols = self.img.getShape()
        for i in range(-2,3):
            for j in range(-2,3):
                if x+i >= 0 and x +1 < rows and y + j >=0 and y+j < cols:
                    self.img[x + i][y + j] = np.asarray([155,155,155])
        self.update()
        plt.show()

    def update(self):
        self.drawArea.clear()
        self.drawArea.axis("off")
        self.drawArea.imshow(self.img, cmap='gray', vmin = 0, vmax = 255)
        
class Searcher():
    def __init__(self, img, drawArea, imgArea):
        self.lcomponents = []
        self.img = Image(img,imgArea)
        self.threshold = 0
        self.rows, self.cols = self.img.getShape()
        self.drawArea = drawArea

    def updateImg(self,img):
        self.img.setImage(img)
        self.img.update()

    def updateThreshold(self,threshold):
        self.threshold = threshold
        self.removeComponents()
    
    def resetImg(self):
        self.img.resetImg()
        reset2('eu')
        self.findComponents(self.lastLabel)
        self.updateText(-1)
    
    def getTotalComponents(self):
        return len(self.lcomponents)
    
    def removeComponents(self):
        for i in range(0,7):
            for c in self.lcomponents:
                d = float(c.getDiameter()) 
                if d < self.threshold:
                    self.img.removePixels(c.lpixels)
                    self.lcomponents.remove(c)

                    
        self.img.update()
        self.updateText(-1)
    
    def searchAround(self,i,j,value,component, obj):
        self.flags[i][j] = obj
        component.add((i,j))

        neighbours = 0
        
        for c in range(-1,2):
            for r in range(-1,2):
                if (j + c) >= 0 and (j + c) < self.cols and (i + r) >= 0 and (i + r) < self.rows:    
                    if(self.img.getPixel(i+r,j+c) == value and self.flags[i+r][j+c] == -1):
                        component = self.searchAround(i+r, j+c, value, component, obj)
                    elif self.img.getPixel(i+r,j+c) != value:
                        neighbours += 1

        if neighbours > 0:
            component.addB((i,j))

        return component
    

    def updateText(self, comp):
        self.drawArea.clear()
        self.drawArea.axis("off")
        text = " Encontrados: %d" % self.getTotalComponents()
        if comp > -1:
            c = self.lcomponents[comp]
            text = text + "\n\nComponente  %d\nDiameter: %.2f\nCentroid(y,x): %.2f-%.2f\nAngle = %.2f" % (comp, c.getDiameter(),c.centroid[1],c.centroid[0],c.teta * 180/math.pi)
        self.drawArea.text(0.5, 0.5, text, va="center", ha="center")
        plt.show()

    def findComponents(self,label):
        self.lastLabel = label
        self.flags = np.full((self.rows,self.cols), -1)
        self.lcomponents = []
   
        if label == 'black':
            value = 0
        else:
            value = 255
        obj = 0
        for i in range(0,self.rows):
            for j in range(0,self.cols):
                if self.img.getPixel(i,j) == value and self.flags[i][j] == -1:
                    component = Component(self.img)
                    component = self.searchAround(i, j, value, component,obj)
                    component.diameter = component.calculateDiameter()
                    component.findCentroid()
                    # component.area = component.calulateArea()
                    # component.perimeter = component.calculatePerimeter()
                    self.lcomponents.append(component)
                    obj += 1
        
        self.updateText(-1)

def update(val):
    global slthres, gray_img , thresh, a0, s, radio_val
    thres = slthres.val
    ret,thresh = cv2.threshold(gray_img,thres,255,cv2.THRESH_BINARY)
    s.updateImg(thresh)
    s.findComponents(radio_val)

def updateE(val):
    global slelmin , s
    s.updateThreshold(slelmin.val)

def find(label):
    global radio_val
    radio_val = label
    s.findComponents(radio_val)

def reset(event):
    global slthres
    slthres.reset()
                                        
def reset2(event):
    global slelmin
    slelmin.reset()

def reset3(event):
    global s
    s.resetImg()

def clickImg(event):
    global a0,s,f,bb
    imSource.resetImg()
    try:
        if event.inaxes is not None:
            ax = event.inaxes
            x,y = f.transFigure.inverted().transform((event.x,event.y))
            if bb.contains(x,y):
                x = int(round(event.xdata))
                y = int(round(event.ydata))
                comp = s.flags[y][x]
                if comp > -1:
                    s.updateText(comp)      
                    imSource.makePoint(s.lcomponents[comp].centroid[1],s.lcomponents[comp].centroid[0])
                    imSource.update()
                    plt.show()                 
    except:
        pass
    # you now have the axes object for that the user clicked on
    # you can use ax.children() to figure out which img artist is in this
    # axes and extract the data from it
    
def clickImg2(event):
    global d0,s,f,bb2
    imSource.resetImg()
    try:
        if event.inaxes is not None:
            ax = event.inaxes
            x,y = f.transFigure.inverted().transform((event.x,event.y))
            if bb2.contains(x,y):
                x = int(round(event.xdata))
                y = int(round(event.ydata))
                comp = s.flags[y][x]
                if comp > -1:
                    s.updateText(comp)      
                    imSource.makePoint(x,y)
                    imSource.update()
                    plt.show()                 
    except:
        pass

def main():
    global slthres, gray_img, thresh, a0, s, radio_val, slelmin,bb,f,source,imSource,bb2
    sys.setrecursionlimit(100000)

    img = cv2.imread('collored2.png')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    radio_val = 'black'

    o = plt.figure()
    gs2 = gridspec.GridSpec(10,10)
    gs2.update( hspace=1.5, wspace=1.5)
    d0 = plt.subplot(gs2[:,: ])
    d0.axis("off")
    plt.get_current_fig_manager().window.wm_geometry("-1200-500")
    d0.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    imSource = Image(img,d0)
    bb2 = d0.get_position()
    f = plt.figure()
    gs = gridspec.GridSpec(11,10)
    gs.update( hspace=1.5, wspace=1.5)
    a0 = plt.subplot(gs[:-2,: -2])
    a1 = plt.subplot(gs[9,:7])
    a2 = plt.subplot(gs[:2, 8:10])
    a3 = plt.subplot(gs[9, 8:10])
    a4 = plt.subplot(gs[2:5, 8:10])
    a5 = plt.subplot(gs[10,:7])
    a6 = plt.subplot(gs[10, 8:10])
    a7 = plt.subplot(gs[8, 8:10])
    plt.get_current_fig_manager().window.wm_geometry("-500-500")

    a0.axis("off")
    a4.axis("off")

    cid = f.canvas.mpl_connect('button_press_event', clickImg)
    cid2 = o.canvas.mpl_connect('button_press_event', clickImg2)

    ret,thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY)
    a0.imshow(thresh, cmap = 'gray',  vmin = 0, vmax = 255)
    bb=a0.get_position()
    
    slthres = Slider(a1,'Threshold', valmin=0,valmax=255, valinit=100, valfmt= "%3.0f")
    slelmin = Slider(a5,'Eliminate', valmin=0,valmax=100, valinit=0, valfmt= "%3.0f")
    radio = RadioButtons(a2,('black', 'white'))
    button = Button(a3,'Reset', color='gray', hovercolor='0.8')
    button2 = Button(a6,'Reset', color='gray', hovercolor='0.8')
    button3 = Button(a7,'Reset Img', color='gray', hovercolor='0.8')
    
    button3.on_clicked(reset3)
    button2.on_clicked(reset2)
    button.on_clicked(reset)
    slthres.on_changed(update)
    slelmin.on_changed(updateE)
    radio.on_clicked(find)
    s = Searcher(thresh, a4, a0)
    update(100)
    s.findComponents(radio_val)
    plt.show()

if __name__ == "__main__":
    main()



