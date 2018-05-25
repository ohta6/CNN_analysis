# encode:utf-8

from chainer.links import VGG16Layers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pickle

model = VGG16Layers()
convs = []
for links in model.links():
    print(dir(links))
    convs.append(links.conv1_1.W.data)
    convs.append(links.conv1_2.W.data)
    convs.append(links.conv2_1.W.data)
    convs.append(links.conv2_2.W.data)
    convs.append(links.conv3_1.W.data)
    convs.append(links.conv3_2.W.data)
    convs.append(links.conv3_3.W.data)
    convs.append(links.conv4_1.W.data)
    convs.append(links.conv4_2.W.data)
    convs.append(links.conv4_3.W.data)
    convs.append(links.conv5_1.W.data)
    convs.append(links.conv5_2.W.data)
    convs.append(links.conv5_3.W.data)
    break
for i,conv in enumerate(convs):
    with open('conv_{}.pickle'.format(i), mode='wb') as f:
        pickle.dump(conv,f)
# link1.shape = (n, 9)となるようにreshape
"""
for fmap in link1:
    link1_reshape = np.concatenate(link1_reshape, fmap.reshape((-1,9))
    map_num.append(i)
"""
# link1_reshape = np.reshape(link1, (-1,9))
#link2_reshape = np.reshape(link2, (-1,9))
#link1_reduced = TSNE(n_components=2, random_state=0).fit_transform(link1_reshape)

#link2_reduced = TSNE(n_components=2, random_state=0).fit_transform(link2_reshape)
#plt.scatter(link1_reduced[:,0], link1_reduced[:,1], s=1)
#plt.savefig('conv1_3.png')
#plt.scatter(link2_reduced[:,0], link2_reduced[:,1])
#plt.savefig('conv1_2.png')
