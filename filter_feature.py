# encode:utf-8

from chainer.links import VGG16Layers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pickle

with open('./conv_weights/conv_0.pickle', mode='rb') as f:
    conv1 = pickle.load(f)

counter = 0
for fmap in conv1:
    for i,layer in enumerate(fmap):
        counter += 1
        xs = np.arange(3)
        z = np.array([layer[x,y] for x in xs for y in xs]).reshape((3,3))
        plt.subplot(3,3,counter)
        xs = np.arange(4)
        plt.pcolor(xs,xs,z)
    if counter == 9:
        break
plt.savefig("layer1.png")
