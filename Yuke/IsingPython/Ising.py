import torch
import cv2


class Ising:
    
    def __init__(self, size_x, size_y, J=1, T=1, K=1):
        self.grid = (torch.randint(0, 2, [1, 1, size_x, size_y]) * 2 - 1).float()
        self.J = J
        self.T = T
        self.K = K

    def deltaE0(self):
        kernel = torch.tensor([[0., 1, 0], [1, 0, 1], [0, 1, 0]]).view((1, 1, 3, 3))
        return torch.conv2d(self.grid, -self.J * 2 * kernel, padding=1) * self.grid

    def step(self):
        prob = torch.exp(-self.deltaE0() / (self.K * self.T))
        flip = (torch.rand_like(prob) < prob) * (torch.rand_like(prob) < 0.1)
        self.grid = self.grid * (flip * (-2) + 1)

    def draw(self, window="Ising"):
        img = self.grid.view(self.grid.size(2), -1).numpy()
        ratio = 600/self.grid.size(2)
        img = cv2.resize(img, (600, (int)(ratio * self.grid.size(3))))
        cv2.imshow(window, img)
        cv2.waitKey(1)
    