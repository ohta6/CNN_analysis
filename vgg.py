# encode:utf-8

from chainer.links import VGG16Layers

model = VGG16Layers()
for links in model.links():
    print(links.conv1_1.W.data.shape)
    print(links.conv1_2.W.data.shape)
    break
