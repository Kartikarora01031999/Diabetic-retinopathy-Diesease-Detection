from scipy import misc
from PIL import Image
from skimage import exposure
from sklearn import svm

import scipy
from math import sqrt,pi
from numpy import exp
from matplotlib import pyplot as plt
import numpy as np
import glob
import matplotlib.pyplot as pltss
import cv2
from matplotlib import cm
import pandas as pd
from math import pi, sqrt
import pywt 
#img_rows=img_cols=200
img_matrix=[]
img_unpred = []
#image_path = Image.open('C:\Users\Rohan\Desktop\Diabetic_Retinopathy\diaretdb1_v_1_1\diaretdb1_v_1_1\resources\images\ddb1_fundusimages\image0')
#image = misc.imread(image_path)

for i in range(1,90):
    img_pt = r'C:\Users\user\Desktop\Diabetic-Retinopathy-Detection-using-CNN-master\diaretdb1_v_1_1\resources\images\ddb1_fundusimages\image'
    if i < 10:
        img_path = img_path + "00" + str(i) + ".png"
    else:
        img_path = img_path + "0" + str(i)+ ".png"

    img = cv2.imread(img_path)
    
    #im_unpre.append(np.array(img).flatten())
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_gray) 
    img_matrix.append(np.array(equ).flatten())
    #res = np.hstack((img_gray,equ))
    
    np.shape(np.array(equ).flatten())
    #print(np.shape(np.array(equ).flatten()))

np.shape(img_matrix)
np.shape(equ)
# plt.imshow(immatrix[78].reshape((1152,1500)),cmap='gray')
# plt.show()
img_dwt = []
for equ in img_matrix:
    equ = equ.reshape((1152,1500))
    cof = pywt.dwt2(equ, 'haar')
    equ2 = pywt.idwt2(cof, 'haar')
    img_dwt.append(np.array(equ2).flatten())
# np.shape(img_dwt)
np.shape(equ2)
plt.imshow(img_dwt[78].reshape((1152,1500)),cmap='gray')
plt.show()
def filter_kernel(L, sigma, t = 3, mf = True):
    dim_y = int(L)
    dim_x = 2 * int(t * sigma)
    arr = np.zeros((dim_y, dim_x), 'f')
    
    ctrl_x = dim_x / 2 
    ctrl_y = int(dim_y / 2.)

    # an un-natural way to set elements of the array
    # to their x coordinate. 
    # x's are actually columns, so the first dimension of the iterator is used
    itera = np.nditer(arr, flags=['multi_index'])
    while not itera.finished:
        arr[itera.multi_index] = itera.multi_index[1] - ctrl_x
        itera.iternext()

    sigma_sq = 2 * sigma * sigma
    sigma_sqrt = 1. / (sqrt(2 * pi) * sigma)
    if not mf:
       sigma_sqrt = sigma_sqrt / sigma ** 2

    #@vectorize(['float32(float32)'], target='cpu')
    def k_fun(x):
        return sigma_sqrt * exp(-x * x / sigma_sq)

    #@vectorize(['float32(float32)'], target='cpu')
    def k_fun_derivative(x):
        return -x * sigma_sqrt * exp(-x * x / sigma_sq)

    if mf:
        kernel = k_fun(arr)
        kernel = kernel - kernel.mean()
    else:
        kernel = k_fun_derivative(arr)

    # return the "convolution" kernel for filter2D
    return cv2.flip(kernel, -1) 

def show_images(images,titles=None, scale=1.3):
    """Display a list of images"""
    num_img = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,num_img + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        figure = fig.add_subplot(1,num_img,n) # Make subplot
        if image.ndim == 2: # Is image grayscale?
            plt.imshow(image, cmap = cm.Greys_r)
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        figure.set_title(title)
        plt.axis("off")
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches(), dtype=np.float) * num_img / scale)
    plt.show()


def gaussian_kernel_matched(L, sigma, t = 3):
    '''
    K =  1/(sqrt(2 * pi) * sigma ) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
    '''
    return filter_kernel(L, sigma, t, True)

#Creating a matched filter bank using the kernel generated from the above functions
def createFilterMatch(K, n = 12):
    rotate = 180 / n
    center = (K.shape[1] / 2, K.shape[0] / 2)
    cur_rotate = 0
    kernels = [K]

    for i in range(1, n):
        cur_rotate += rotate
        rot_mat = cv2.getRotationMatrix2D(center, cur_rotate, 1)
        k = cv2.warpAffine(K, rot_mat, (K.shape[1], K.shape[0]))
        kernels.append(k)

    return kernels

#Given a filter bank, apply them and record maximum response

def applyFilters(im, kernels):

    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)


gf = gaussian_kernel_matched(20, 5)
bank_gf = createFilterMatch(gf, 4)

img_gaus = []
for equ2 in img_dwt:
    equ2 = equ2.reshape((1152,1500))
    equ3 = applyFilters(equ2,bank_gf)
    img_gaus.append(np.array(equ3).flatten())
np.shape(img_gaus)
plt.imshow(img_gaus[78].reshape((1152,1500)),cmap='gray')
plt.show()
def createFilterMatch():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 6, theta,12, 0.37, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def applyFilters(im, kernels):
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)

bank_gf = createFilterMatch()
#equx=equ3
#equ3 = applyFilters(equ2,bank_gf)
img_gaus2 = []
for equ2 in img_dwt:
    equ2 = equ2.reshape((1152,1500))
    equ3 = applyFilters(equ2,bank_gf)
    img_gaus2.append(np.array(equ3).flatten())
np.shape(img_gaus2)
plt.imshow(img_gaus2[20].reshape((1152,1500)),cmap='gray')
plt.show()
np.shape(img_gaus2)
plt.imshow(img_gaus2[1].reshape((1152,1500)),cmap='gray')
plt.show()
e = equ3
np.shape(e)
e=e.reshape((-1,3))
np.shape(e)

img = equ3
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

k=cv2.KMEANS_PP_CENTERS


# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
img_kmean = []
for equ3 in img_gaus2:
    img = equ3.reshape((1152,1500))
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    k=cv2.KMEANS_PP_CENTERS


    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    img_kmean.append(np.array(res2).flatten())
np.shape(img_kmean)

plt.imshow(img_kmean[78].reshape((1152,1500)),cmap="gray")
plt.show()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC()
Y = np.ones(89)

Y[1]=Y[5]=Y[7]=Y[17]=Y[6]=0

print(clf.fit(img_kmean, Y))



y_pred = clf.predict(img_kmean)

#k = [1,3,4,9,10,11,13,14,20,22,24,25,26,27,28,29,35,36,38,42,53,55,57,64,70,79,84,86]


i#mm_train = k-np.ones(len(k))

#print (k)
#clf.fit(imm_train, y_train)
#y_pred = clf.predict(img_kmean)
print(accuracy_score(Y,y_pred))

# kNN
#from sklearn.neighbors import KNeighborsClassifier
#neigh = KNeighborsClassifier(n_neighbors=3)

#neigh.fit(imm_train, y_train)
#y_pred2=neigh.predict(img_kmean)
#print(accuracy_score(Y,ypred2))
#
