# encode:utf-8

from chainer.links import VGG16Layers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

model = VGG16Layers()
for links in model.links():
    link1 = links.conv1_1.W.data
    link2 = links.conv1_2.W.data
    break

# link1.shape = (n, 9)となるようにreshape
link1_reshape = np.reshape(link1, (-1,9))
link2_reshape = np.reshape(link2, (-1,9))
link1_reduced = TSNE(n_components=2, random_state=0).fit_transform(link1_reshape)

link2_reduced = TSNE(n_components=2, random_state=0).fit_transform(link2_reshape)
plt.scatter(link1_reduced[:,0], link1_reduced[:,1])
plt.savefig('conv1_1.png')
#plt.scatter(link2_reduced[:,0], link1_reduced[:,1])
#plt.savefig('conv1_2.png')
