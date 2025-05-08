import numpy as np

conv_result = np.convolve((1, 2, 3), (4, 5, 6))
"""

"""

print(conv_result)

x_1 = np.arange(1, 7)
x_2 = np.arange(1, 7)[::-1]

print(np.convolve(x_1, x_2))