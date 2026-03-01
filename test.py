# load indicator.npy
import numpy as np
import matplotlib.pyplot as plt
indicator = np.load("indicator.npy")

plt.imshow(indicator, cmap='gray')
plt.title("Indicator")
plt.colorbar()
plt.show()
