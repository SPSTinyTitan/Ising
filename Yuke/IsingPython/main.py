import Ising
import cv2
import torch
import time

iters = 100
window = 1000
#model = Ising.Ising(100, 100, J=-1, T=0.2, K=1)
model = Ising.Ising3D(100, 100, 100, J=-1, T=0.2, K=1)
window = "Ising"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)


y = torch.empty(iters)
t = time.perf_counter()
for i in range(iters):
    y[i] = model.step()
    print(y[i])
    model.draw(window)

print("Time elapsed", time.perf_counter() - t)