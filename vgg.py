# encode:utf-8

from chainer.links import VGG16Layers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

model = VGG16Layers()
for links in model.links():
    link1 = links.conv1_1.W.data.shape
    print(links.conv1_2.W.data.shape)
    break

# link1.shape = (n, 9)となるようにreshape
link1_reshape = link1.reshape((-1,9))
link1_reduced = TSNE(n_components=2, random_state=0).fit_transform(link1_reshape)

plt.scatter(link1_reduced[:,0], link1_reduced[:,1])
