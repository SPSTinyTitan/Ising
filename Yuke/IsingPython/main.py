import Ising
import cv2
model = Ising.Ising(256, 256, J=1, T=0.2, K=1)
window = "Ising"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

for i in range(1000):
    model.step()
    model.draw(window)