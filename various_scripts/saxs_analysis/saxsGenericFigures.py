import numpy as np
import matplotlib.pyplot as plt


r = np.array(
   [np.random.normal(m, s, 10000) for m, s in zip([0, 7, 11, 22], [0.2, 0.5, 1, 3])]
) 
angle = np.linspace(0, 2 * np.pi , 10000)

X = [r[0] * np.cos(angle), r[1] * np.cos(angle), r[2] * np.cos(angle), r[3] * np.cos(angle)]
Y = [r[0] * np.sin(angle), r[1] * np.sin(angle), r[2] * np.sin(angle), r[3] * np.sin(angle)]

fig, ax = plt.subplots(figsize = (10,10))
# ax.axis('off')
ax.set_facecolor('darkblue')

ax.scatter(X[0], Y[0], c='magenta', alpha = 0.2)
ax.scatter(X[1], Y[1], c='orange', alpha = 0.2)
ax.scatter(X[2], Y[2], c='red', alpha = 0.2)
ax.scatter(X[3], Y[3], c='purple', alpha = 0.2)

ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
