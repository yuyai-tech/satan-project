
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from PIL import Image

# one y one
file_name = '/ann_advanced/png_images/00.png'
img = np.array(Image.open(file_name))
img_binary = []
for i in img:
    row = []
    for j in i:
        row.append(int(j[3]/255))
    img_binary.append(row)

data_frame = pd.DataFrame(img_binary)

plt.imshow(data_frame)
plt.show()


# all
index = ['0', '1', '2', '3', 'w']
for idx in index:
    print(idx)
    for num in range(0, 5):
        file_name = '/Users/jnfran92/PycharmProjects/satan-project/ann_advanced/png_images/%s%d.png' % (idx, num)
        print(file_name)

        img = np.array(Image.open(file_name))
        img_binary = []
        for i in img:
            row = []
            for j in i:
                row.append(int(j[3] / 255))
            img_binary.append(row)

        data_frame = pd.DataFrame(img_binary)
        #
        # plt.imshow(data_frame)
        # plt.show()

        data_folder = '/Users/jnfran92/PycharmProjects/satan-project/ann_advanced/dataset' + '/' + idx
        file_name = str(random.randint(0, 10 ** 15))
        file_path = data_folder + '/' + file_name + '.csv'
        data_frame.to_csv(file_path, header=False, index=False)
