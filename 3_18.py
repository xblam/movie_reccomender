import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# this will be our function
f = lambda x: np.exp(x)
df = lambda x: np.exp(x)

# points to use during spline
x = np.array([0, 1, 2])
y = f(x)

# clamped first df at endpoints
bc_type = ((1, df(0)), (1, df(2)))

# make cubic spline
spline = CubicSpline(x, y, bc_type=bc_type)

# evaluate the spline
x_dense = np.linspace(0, 2, 200)
y_true = f(x_dense)
y_spline = spline(x_dense)

# plot all 
plt.plot(x_dense, y_true, label='f(x) = e^x', linestyle='--')
plt.plot(x_dense, y_spline, label='Clamped Cubic Spline $S_c$')
plt.scatter(x, y, color='red', zorder=5)
plt.legend()
plt.grid(True)
plt.title("Clamped Cubic Spline Approximation of $f(x) = e^x$")
plt.xlabel("x")
plt.ylabel("y")
plt.show()