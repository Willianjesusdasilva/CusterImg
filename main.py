#https://i.imgur.com/PLGlnWj.png
#https://images.vexels.com/media/users/3/153996/raw/6032522ba9ee18fa9afa69ed780b7c02-bocha-bolas-coloridas-bola-conjunto-grafico.jpg

import numpy as np
import urllib
import cv2
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv

url = f'https://i.imgur.com/PLGlnWj.png'

img = url_to_image(url)
imgTemplate = img.copy()
BLUE_MIN = (110,50,50)
BLUE_MAX = (130,255,255)

data_cord = []

height, width, channels = img.shape
size = height,width

for x in range(size[0]):
    
    for y in range(size[1]):
        r, g, b = img[x,y]
        if (r,g,b) >= BLUE_MIN and (r,g,b) <= BLUE_MAX:
            data_cord.append([int(x),int(y)])
            img[x,y] = (255,0,0)
        else: 
            img[x,y] = (0,0,0)

cv2.imshow('a',img)

X = StandardScaler().fit_transform(data_cord)

db = DBSCAN(eps=0.1, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

import matplotlib.pyplot as plt

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        pass

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)
    
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

dict_data = dict(Counter(labels))
template_max = max(dict_data, key=dict_data.get)
result_cluster = []

for i in range(len(labels)):
    if labels[i] == template_max:
      x,y = data_cord[i]
      result_cluster.append(data_cord[i])
      r, g, b = imgTemplate[x,y]
      img[x,y] = (r,g,b)
    else:
      x,y = data_cord[i]
      img[x,y] = (0,0,0)

imgFinal = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
cv2.imshow('a',imgFinal)
cv2.waitKey()