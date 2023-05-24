import numpy as np
import matplotlib.pyplot as plt
import pandas
from PIL import Image,ImageTk
import cv2
from tkinter import filedialog
from pulp import *
import pandas as pd
import tkinter as tk
from tkinter import Button,Label
import math
from tkinter import ttk

from openpyxl import load_workbook

root = tk.Tk()
root.geometry("1000x600")
root.title("image processor")
root.config(bg="black")
file_path=""
imgstandby=cv2.imread("pictures/waiter.jpg",0)

#imstand=Image.fromarray(imgstandby)
#imgtk=ImageTk.PhotoImage(image=imstand)
def unsharp_masking_and_Highboost_masking(img, kernel_size,const):
    padd = kernel_size // 2
    height, width = img.shape[:2]
    new_height = height + 2 * padd
    new_width = width + 2 * padd
    new_image = np.zeros((new_height, new_width), dtype=img.dtype)
    i = padd

    for r in range(padd, height + padd):
        j = padd
        for c in range(padd, width + padd):
            new_image[i, j] = img[r - padd, c - padd]
            j += 1
        i += 1

    #Average --> Blured Image
    avg_img = np.zeros((new_height,new_width))

    for o in range(padd,new_height - padd):
      for l in range(padd,new_width - padd):
        tot = 0
        for d in range(-padd,padd+1):
          for f in range(-padd,padd+1):
            tot += new_image[o+d,l+f]
        avg_img[o,l] = tot / (kernel_size ** 2)

    avg_img = avg_img[padd:height+padd, padd:width+padd]

    unsharp_mask = img - avg_img
    sharp_img = img + const * unsharp_mask


    return sharp_img , avg_img

def laplacianscaling(img,scale,kernel_size):
    padd = kernel_size // 2
    height, width = img.shape[:2]
    new_height = height + 2 * padd
    new_width = width + 2 * padd

    laplaimg = np.zeros((new_height, new_width), dtype=img.dtype)
    i = padd

    for r in range(padd, height + padd):
        k = padd
        for c in range(padd, width + padd):
            laplaimg[i, k] = img[r - padd, c - padd]
            k += 1
        i += 1
    output2 = np.zeros((height, width), dtype=img.dtype)
    for R in range(padd, height + padd):
        for C in range(padd, width + padd):
            dx = laplaimg[R,C+1]-2*laplaimg[R,C]+laplaimg[R,C-1]
            dy = laplaimg[R+1,C]-2*laplaimg[R,C]+laplaimg[R-1,C]
            output2[R - padd, C - padd] = dx + dy
    output2 = output2 * scale
    min = np.min(output2)
    max = np.max(output2)
    output2 = ((output2 - min) / (max - min)) * 255

    return output2

def laplacian(img, kernel_size):
    padd = kernel_size // 2
    height, width = img.shape[:2]
    new_height = height + 2 * padd
    new_width = width + 2 * padd
    new_image = np.zeros((new_height, new_width), dtype=img.dtype)
    i = padd

    for r in range(padd, height + padd):
        k = padd
        for c in range(padd, width + padd):
            new_image[i, k] = img[r - padd, c - padd]
            k += 1
        i += 1


    output = np.zeros((height, width), dtype=img.dtype)
    for x in range(padd, height + padd):
        for y in range(padd, width + padd):
            dx = new_image[x, y + 1] - 2 * new_image[x, y] + new_image[x, y - 1]
            dy = new_image[x + 1, y] - 2 * new_image[x, y] + new_image[x - 1, y]
            output[x - padd, y - padd] = dx + dy

    return output

def sobel(img,kernel_size):
    padd = kernel_size // 2
    height, width = img.shape[:2]
    new_height = height + 2 * padd
    new_width = width + 2 * padd
    new_image = np.zeros((new_height, new_width), dtype=img.dtype)
    i = padd

    for r in range(padd, height + padd):
        k = padd
        for c in range(padd, width + padd):
            new_image[i, k] = img[r - padd, c - padd]
            k += 1
        i += 1


    output6 = np.zeros((height, width), dtype=img.dtype)
    output7 = np.zeros((height, width), dtype=img.dtype)
    output8 = np.zeros((height, width), dtype=img.dtype)
    kernel_H = [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]


    kernel_V = [[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]]


    for h in range(1, height - 1):
      for v in range(1, width - 1):
        sh = 0
        sv = 0
        for ind in range(-1, 2):
          for jnd in range(-1, 2):
            sh += kernel_H[ind + 1][jnd + 1] * new_image[h + ind, v + jnd]
            sv += kernel_V[ind + 1][jnd + 1] * new_image[h + ind, v + jnd]
        output6[h, v] = sh
        output7[h, v] = sv
        output8[h, v] = sh + sv

    return output6,output7,output8

def adder():
    print("hello")
    global file_path
    file_path = filedialog.askopenfilename(initialdir="pictures")
    img=cv2.imread(file_path,0)
    cv2.imshow("hello",img)
    im=Image.fromarray(img)
    global imgtk
    imgtk=ImageTk.PhotoImage(image=im)
    global imgnoise
    imgnoise=cv2.imread(file_path,0)

def average():
    print("im here")
    img = cv2.imread(file_path, 0)
    imgaverage= cv2.imread(file_path, 0)

    for r in range(img.shape[0] - 1):
        for c in range(img.shape[1] - 1):
            lm = 0
            if r > 0 and c > 0:
                n1 = img[r - 1, c - 1]
                n2 = img[r - 1, c]
                n3 = img[r - 1, c + 1]
                n4 = img[r, c - 1]

                n6 = img[r, c + 1]
                n7 = img[r + 1, c - 1]
                n8 = img[r + 1, c]
                n9 = img[r + 1, c + 1]

                lm = lm + n1
                lm = lm + n2
                lm = lm + n3
                lm = lm + n4

                lm = lm + n6
                lm = lm + n7
                lm = lm + n8
                lm = lm + n9
                lm = lm / 8

                imgaverage[r, c] = lm
    cv2.imshow("averaged",imgaverage)
    im = Image.fromarray(imgaverage)

    imgtk = ImageTk.PhotoImage(image=im)

def median():
    img = cv2.imread(file_path, 0)
    imgmedian = cv2.imread(file_path, 0)

    for r in range(img.shape[0] - 1):
        for c in range(img.shape[1] - 1):
            n1 = img[r - 1, c - 1]
            n2 = img[r - 1, c]
            n3 = img[r - 1, c + 1]
            n4 = img[r, c - 1]
            n5 = img[r, c]
            n6 = img[r, c + 1]
            n7 = img[r + 1, c - 1]
            n8 = img[r + 1, c]
            n9 = img[r + 1, c + 1]
            median = [n1, n2, n3, n4, n5, n6, n7, n8, n9]
            median.sort()
            imgmedian[r, c] = median[4]
    median
    cv2.imshow("median",imgmedian)

def adaptive_median():
    s = 3
    sMax = 7
    img=cv2.imread(file_path,0)
    kernel_w = s
    kernel_h = s

    st = 0

    img_w = img.shape[0]
    img_h = img.shape[1]

    pad = np.zeros((img_w + kernel_w // 2, img_h + kernel_h // 2))
    pad[kernel_w // 2: img_w + kernel_w // 2, kernel_h // 2: img_h + kernel_h // 2] = img.copy()

    new_img = np.zeros((img_w, img_h))

    for i in range(st, pad.shape[0] - kernel_w + 1):
        for j in range(st, pad.shape[1] - kernel_h + 1):
            matrix = pad[i: i + kernel_w, j: j + kernel_h]
            val = list(matrix.flatten())
            zmed = np.median(val)
            zmax = np.max(val)
            zmin = np.min(val)
            zxy = img[i, j]
            a1 = zmed - zmin
            a2 = zmed - zmax

            if (a1 > 0 and a2 < 0):
                b1 = zxy - zmin
                b2 = zxy - zmax
                if (b1 > 0 and b2 < 0):
                    new_img[i, j] = zxy
                else:
                    new_img[i, j] = zmed
            elif (st != sMax):
                s = s + 1
                st = 0
            elif (st >= sMax):
                new_img[i, j] = zmed
    cv2.imshow("adaptive median",new_img)

def forwardfourier():
    img=cv2.imread(file_path,0)
    height, width = 50,50

    complexer = np.zeros((height, width), dtype=complex)

    for i in range(height):
        for j in range(width):
            complexer[i, j] = img[i, j] * np.exp(-1j * 2 * np.pi * (i / height) * (j / width))

    fourierforward = np.empty((height, width), dtype=complex)
    for u in range(height):
        for v in range(width):
            fourierforward[u, v] = 0
            for i in range(height):
                for j in range(width):
                    fourierforward[u, v] += complexer[i, j] * np.exp(
                        -1j * 2 * np.pi * ((i * u / height) + (j * v / width)))

    for i in range(height):
        for j in range(width):
            fourierforward[i,j]=float(abs(fourierforward[i,j]))
    cv2.imshow("forward",fourierforward)

def guassianfilter():
    print("heyo")
    imgguass = cv2.imread(file_path, 0)
    imgreal = cv2.imread(file_path, 0)
    for r in range(imgreal.shape[0] - 1):
        for c in range(imgreal.shape[1] - 1):

            if r > 0 and c > 0:
                n1 = imgreal[r - 1, c - 1]
                n2 = imgreal[r - 1, c]
                n3 = imgreal[r - 1, c + 1]
                n4 = imgreal[r, c - 1]
                n5 = imgreal[r, c]
                n6 = imgreal[r, c + 1]
                n7 = imgreal[r + 1, c - 1]
                n8 = imgreal[r + 1, c]
                n9 = imgreal[r + 1, c + 1]
                n1 *= 1
                n2 *= 2
                n3 *= 1
                n4 *= 2
                n5 *= 4
                n6 *= 2
                n7 *= 1
                n8 *= 2
                n9 *= 1
                imgguass[r, c] = (n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9) / 16
    cv2.imshow("guassian",imgguass)

def nearest_neighbor_interpolation():
    print("hey")
    image=cv2.imread(file_path,0)
    new_height=int(input())
    new_width=int (input())

    new_image = np.zeros((new_height, new_width), dtype=np.uint8)

    scale_x = float(image.shape[1]) / new_width
    scale_y = float(image.shape[0]) / new_height

    for y in range(new_height):
        for x in range(new_width):
            px = int(x * scale_x)
            py = int(y * scale_y)

            if px >= image.shape[1]:
                px = image.shape[1] - 1
            if py >= image.shape[0]:
                py = image.shape[0] - 1

            new_image[y, x] = image[py, px]
    cv2.imshow("nearest neighbor",new_image)

def bilinear_interpolation():
    image=cv2.imread(file_path)
    new_height=int(input())
    new_width = int(input())
    scale_x = float(image.shape[1]) / new_width
    scale_y = float(image.shape[0]) / new_height

    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            px = x * scale_x
            py = y * scale_y

            x1, y1 = int(px), int(py)
            x2, y2 = x1 + 1, y1 + 1

            if x2 >= image.shape[1]:
                x2 = x1
            if y2 >= image.shape[0]:
                y2 = y1

            tl = image[y1, x1]
            tr = image[y1, x2]
            bl = image[y2, x1]
            br = image[y2, x2]

            qx = px - x1
            qy = py - y1

            top = tl * (1 - qx) + tr * qx

            bottom = bl * (1 - qx) + br * qx

            interpolated = top * (1 - qy) + bottom * qy

            new_image[y, x] = interpolated
    cv2.imshow("bilinear",new_image)

def fouriertrick():
    image=cv2.imread(file_path)
    new_height=int(input())
    new_width = int(input())
    scale_x = float(image.shape[1]) / new_width
    scale_y = float(image.shape[0]) / new_height

    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            px = x * scale_x
            py = y * scale_y

            x1, y1 = int(px), int(py)
            x2, y2 = x1 + 1, y1 + 1

            if x2 >= image.shape[1]:
                x2 = x1
            if y2 >= image.shape[0]:
                y2 = y1

            tl = image[y1, x1]
            tr = image[y1, x2]
            bl = image[y2, x1]
            br = image[y2, x2]

            qx = px - x1
            qy = py - y1

            top = tl * (1 - qx) + tr * qx

            bottom = bl * (1 - qx) + br * qx

            interpolated = top * (1 - qy) + bottom * qy

            new_image[y, x] = interpolated

def histonorm():
    img = cv2.imread(file_path,0)

    img.shape
    size = img.shape[0] * img.shape[1]
    size
    img2 = img.copy()
    img3 = img.copy()
    frequency = [0.0] * 256
    print(frequency)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            frequency[img[r, c]] += 1
    print(frequency)
    probability = frequency.copy()
    i = 0
    while (i < 256):
        probability[i] = probability[i] / 129600
        i += 1
    print(probability)
    normalized = [0]*256
    i = 0
    sum = 0
    while (i < 256):
        sum += probability[i]
        normalized[i] += 255
        sum
        i += 1
    print(normalized)
    i = 0
    while (i < 256):
        normalized[i] = np.round(normalized[i])
        i += 1
    print(normalized)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            img2[r, c] = normalized[img[r, c]]
    cv2.imshow("img1",img)
    cv2.imshow("img2",img2)
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.show()
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.show()
    img3 = cv2.equalizeHist(img)
    cv2.imshow("img2",img2)
    cv2.imshow("img3",img3)

def robert():
    img=cv2.imread(file_path,0)
    RCGO_X, RCGO_Y, Final_RCGO = Cross_Gradient_Operators(img, kernel_size=3)
    cv2.imshow("rcgo_X",RCGO_X)
    cv2.imshow("rcgo_y", RCGO_Y)
    cv2.imshow("final", Final_RCGO)

def sobel1():
    img=cv2.imread(file_path,0)
    S_H, S_V, F_S = sobel(img, kernel_size=3)
    cv2.imshow("Horizontal",S_H)
    cv2.imshow("vertical",S_V)
    cv2.imshow("FULL",F_S)

def laplano():
    gray_img=cv2.imread(file_path,0)
    lpo_without_scaling = laplacian(gray_img, kernel_size=3)
    cv2.imshow("laplanoscaling",lpo_without_scaling)

def laplayes():
    img=cv2.imread(file_path,0)
    lpo_with_scaling = laplacianscaling(img, 1, kernel_size=3)
    cv2.imshow("scale laplace",lpo_with_scaling)

def unsharp():
    img=cv2.imread(file_path,0)
    masking, avg = unsharp_masking_and_Highboost_masking(img, 3, 1)
    himasking, avg = unsharp_masking_and_Highboost_masking(img, 3, 4)
    cv2.imshow("highboost",himasking)
    cv2.imshow("unsharp",masking)

def Cross_Gradient_Operators(img,kernel_size):
    padd = kernel_size // 2
    height, width = img.shape[:2]
    new_height = height + 2 * padd
    new_width = width + 2 * padd
    new_image = np.zeros((new_height, new_width), dtype=img.dtype)
    i = padd

    for r in range(padd, height + padd):
        k = padd
        for c in range(padd, width + padd):
            new_image[i, k] = img[r - padd, c - padd]
            k += 1
        i += 1


    output3 = np.zeros((height, width), dtype=img.dtype)
    output4 = np.zeros((height, width), dtype=img.dtype)
    output5 = np.zeros((height, width), dtype=img.dtype)
    for a in range(padd, height + padd):
      for z in range(padd, width + padd):
          dxx = new_image[a,z]-new_image[a+1,z+1]
          dyy = new_image[a+1,z]-new_image[a,z+1]
          output3[a- padd,z- padd] = dxx
          output4[a - padd,z - padd] = dyy
          output5[a - padd,z - padd] = dxx + dyy
    return output3,output4,output5

def uniform_noise():
    temp = []
    global img_uni
    img_uni=imgnoise.copy()
    s = 0
    for r in range(1, imgnoise.shape[0] - 1):
        for c in range(1, imgnoise.shape[1] - 1):
            for i in range(r - 1, r + 1):
                for j in range(c - 1, c + 1):
                    temp.append(imgnoise[i, j])
            max = np.max(temp)
            min = np.min(temp)
            s = (imgnoise[i, j] + (min + (max - min) * np.random.random()))
            img_uni[r, c] = s
            temp.clear()

    print(" ")

    cv2.imshow("uniform",img_uni)
    #meanfixer
    img_Amean = img_uni.copy()
    r = 1
    s = 0
    while r < img_uni.shape[0] - 1:
        c = 1
        while c < img_uni.shape[1] - 1:
            i = r - 1
            j = c - 1
            while i < r + 1:
                while j < c + 1:
                    s += img_uni[i, j]
                    j += 1
                i += 1
            img_Amean[r, c] = s / 9
            s = 0
            c += 1
        r += 1
        #fixing with geo mean
    img_Gmean = img_uni.copy()
    r = 1
    s = 1
    while r < img_uni.shape[0] - 1:
        c = 1
        while c < img_uni.shape[1] - 1:
            s = img_uni[r - 1, c - 1]
            s = img_uni[r - 1, c]
            s = img_uni[r - 1, c + 1]
            s = img_uni[r, c - 1]
            s = img_uni[r, c]
            s = img_uni[r, c + 1]
            s = img_uni[r + 1, c - 1]
            s = img_uni[r + 1, c]
            s *= img_uni[r + 1, c + 1]
            s = s ** (1 / 9)
            img_Gmean[r, c] = s
            s = 1
            c += 1
        r += 1

    img_med = img_uni.copy()
    r = 1
    m = []
    while r < img_uni.shape[0] - 1:
        c = 1
        while c < img_uni.shape[1] - 1:
            m.append(img_uni[r - 1, c - 1])
            m.append(img_uni[r - 1, c])
            m.append(img_uni[r - 1, c + 1])
            m.append(img_uni[r, c - 1])
            m.append(img_uni[r, c])
            m.append(img_uni[r, c + 1])
            m.append(img_uni[r + 1, c - 1])
            m.append(img_uni[r + 1, c])
            m.append(img_uni[r + 1, c + 1])
            m.sort()
            # print(m)
            img_med[r, c] = m[4]
            m.clear()
            # print(m)
            c += 1
        r += 1

    img_mid = img_uni.copy()
    s = 0
    for r in range(1, img_uni.shape[0] - 1):
        for c in range(1, img_uni.shape[1] - 1):
            for i in range(r - 1, r + 1):
                for j in range(c - 1, c + 1):
                    temp.append(img_uni[i, j])
            max = np.max(temp)
            min = np.min(temp)
            s = (max + min) * (1 / 2)
            img_mid[r, c] = s
            temp.clear()

    img_harc = img_uni.copy()
    sum = 0
    val = 0
    for r in range(1, img_uni.shape[0] - 1):
        for c in range(1, img_uni.shape[1] - 1):
            for i in range(r - 1, r + 1):
                for j in range(c - 1, c + 1):
                    sum += 1 / img_uni[i, j]
            val = (9 / sum)
            img_harc[r, c] = val
            sum = 0
    print(" ")
    cv2.imshow("fixed with mid",img_mid)
    cv2.imshow("fixed with med",img_med)
    cv2.imshow("fixed with mean",img_Amean)
    cv2.imshow("fixed with geo mean",img_Gmean)
    cv2.imshow("fixed with harmonic mean",img_harc)

def impulse():
    img = cv2.imread(file_path,0)

    size = img.shape[0] * img.shape[1]
    size1 = img.shape
    img_sap = img.copy()
    pep = 0.005
    salt = 1 - pep
    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
            rndm = np.random.random()
            if rndm < pep:
                img_sap[r, c] = 0
            elif rndm > salt:
                img_sap[r, c] = 255
            else:
                img_sap[r, c] = img[r, c]
    img_Amean = img_sap.copy()
    r = 1
    s = 0
    while r < img_sap.shape[0] - 1:
        c = 1
        while c < img_sap.shape[1] - 1:
            i = r - 1
            j = c - 1
            while i < r + 1:
                while j < c + 1:
                    s += img_sap[i, j]
                    j += 1
                i += 1
            img_Amean[r, c] = s / 9
            s = 0
            c += 1
        r += 1
        # fixing with geo mean
    img_Gmean = img_sap.copy()
    r = 1
    s = 1
    while r < img_sap.shape[0] - 1:
        c = 1
        while c < img_sap.shape[1] - 1:
            s = img_sap[r - 1, c - 1]
            s = img_sap[r - 1, c]
            s = img_sap[r - 1, c + 1]
            s = img_sap[r, c - 1]
            s = img_sap[r, c]
            s = img_sap[r, c + 1]
            s = img_sap[r + 1, c - 1]
            s = img_sap[r + 1, c]
            s *= img_sap[r + 1, c + 1]
            s = s ** (1 / 9)
            img_Gmean[r, c] = s
            s = 1
            c += 1
        r += 1

    img_med = img_sap.copy()
    r = 1
    m = []
    while r < img_sap.shape[0] - 1:
        c = 1
        while c < img_sap.shape[1] - 1:
            m.append(img_sap[r - 1, c - 1])
            m.append(img_sap[r - 1, c])
            m.append(img_sap[r - 1, c + 1])
            m.append(img_sap[r, c - 1])
            m.append(img_sap[r, c])
            m.append(img_sap[r, c + 1])
            m.append(img_sap[r + 1, c - 1])
            m.append(img_sap[r + 1, c])
            m.append(img_sap[r + 1, c + 1])
            m.sort()
            # print(m)
            img_med[r, c] = m[4]
            m.clear()
            # print(m)
            c += 1
        r += 1

    img_mid = img_sap.copy()
    s = 0
    temp=[]
    for r in range(1, img_sap.shape[0] - 1):
        for c in range(1, img_sap.shape[1] - 1):
            for i in range(r - 1, r + 1):
                for j in range(c - 1, c + 1):
                    temp.append(img_sap[i, j])
            max = np.max(temp)
            min = np.min(temp)
            s = (max + min) * (1 / 2)
            img_mid[r, c] = s
            temp.clear()

    img_harc = img_sap.copy()
    sum = 0
    val = 0
    for r in range(1, img_sap.shape[0] - 1):
        for c in range(1, img_sap.shape[1] - 1):
            for i in range(r - 1, r + 1):
                for j in range(c - 1, c + 1):
                    sum += 1 / img_sap[i, j]
            val = (9 / sum)
            img_harc[r, c] = val
            sum = 0
    print(" ")
    cv2.imshow("fixed with mid", img_mid)
    cv2.imshow("fixed with med", img_med)
    cv2.imshow("fixed with mean", img_Amean)
    cv2.imshow("fixed with geo mean", img_Gmean)
    cv2.imshow("fixed with harmonic mean", img_harc)

    cv2.imshow("saltep",img_sap)


left_frame=tk.Frame(root, width=400 , height = 600, bg="white")
left_frame.pack(side="left",fill="y")

right_frame=tk.Frame(root, width=400 , height = 600, bg="white")
right_frame.pack(side="right",fill="y")

image_button= Button(left_frame, text="image add", bg="green")
image_button['command']=adder
image_button.pack(pady=15)
adder()
average_button= Button(left_frame, text="average", bg="pink",width=20)
average_button['command']=average
average_button.pack(pady=15)

median_button= Button(left_frame, text="median", bg="blue",width=20)
median_button['command']=median
median_button.pack(pady=15)

adaptive_button= Button(left_frame, text="adaptive median", bg="purple")
adaptive_button['command']=adaptive_median
adaptive_button.pack(pady=15)

fourierforward= Button(left_frame, text="forward fourier ( takes a long time)", bg="orange")
fourierforward['command']=forwardfourier
fourierforward.pack(pady=15)

guassian= Button(left_frame, text="guassian", bg="indigo")
guassian['command']=guassianfilter
guassian.pack(pady=15)

bilnear= Button(left_frame, text="bilnear", bg="cyan")
bilnear['command']=bilinear_interpolation
bilnear.pack(pady=15)

nearestneighbor= Button(left_frame, text="nearestneighbor", bg="gray")
nearestneighbor['command']=nearest_neighbor_interpolation
nearestneighbor.pack(pady=15)

unsharpsharp = Button(left_frame, text="unsharpsharp/highboost", bg="gray")
unsharpsharp['command']=unsharp
unsharpsharp.pack(pady=15)

roberts= Button(left_frame, text="robert", bg="red")
roberts['command']=robert
roberts.pack(pady=15)

norm = Button(left_frame, text="norm", bg="darkgreen")
norm['command']=histonorm
norm.pack()

uni = Button(right_frame, text="uniformnoise", bg="blue")
uni['command']=uniform_noise
uni.pack(pady=15)

impulsenoise = Button(right_frame, text="impulsenoise", bg="yellow")
impulsenoise['command']=impulse
impulsenoise.pack()

sobeler = Button(right_frame, text="sobel", bg="indigo")
sobeler['command']=sobel1
sobeler.pack(pady=15)

laplaceno = Button(right_frame, text="laplacians no scale", bg="gray")
laplaceno['command']=laplano
laplaceno.pack(pady=15)

laplaceyes = Button(right_frame, text="laplacians scaling", bg="orange")
laplaceyes['command']=laplayes
laplaceyes.pack(pady=15)

imager=Label(root,image=imgtk)
imager.pack()
root.mainloop()