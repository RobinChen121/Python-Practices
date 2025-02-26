import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
plt.plot(x, np.sin(x))
plt.title('My picture')
plt.savefig('MyPicture')
plt.show()


N = 8
y = np.zeros(N)
x1 = np.linspace(0, 10, N, endpoint=True)
x2 = np.linspace(0, 10, N, endpoint=False)
plt.plot(x1, y, 'o')
plt.plot(x2, y, 'o')
plt.show()

plt.plot([1,2])
plt.show()
