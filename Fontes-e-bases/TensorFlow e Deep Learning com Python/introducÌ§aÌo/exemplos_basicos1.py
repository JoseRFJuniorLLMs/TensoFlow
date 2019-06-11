numero = 2

vetor = [2, 3, 5]

import numpy as np
vetor = np.asarray(vetor)

vetor.max()
vetor.argmax()
vetor.min()
vetor.mean()
vetor.sum()

vetor2 = numero * vetor

np.arange(1, 50, 0.5)

np.zeros(10)
np.zeros((3,3))
np.ones(10)
np.ones((3,3))

np.linspace(1, 20, 5)

np.random.seed(1)
np.random.randint(0, 101, 4)

np.random.randint(0, 101, (3,3))