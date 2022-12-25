import matplotlib.pyplot as plt 
import numpy as np
def imshow(img):
    img = img/2 + 0.5  # image unnormalization
    plt.imshow(np.transpose(img,(1,2,0)))
    