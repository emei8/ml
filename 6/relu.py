import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10,10)
y = np.array([0 * item if item < 0 else item for item in x])

plt.figure()
plt.plot(x,y ,label="Relu")
plt.legend()
plt.show()
