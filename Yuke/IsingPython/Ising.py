import torch
import cv2


class Ising:
    
    def __init__(self, size_x, size_y, J=1, T=1, K=1, coarse_size=24):
        self.grid = (torch.randint(0, 2, [1, 1, size_x, size_y]) * 2 - 1).float()
        self.size_x = size_x
        self.size_y = size_y
        self.coarse_size = coarse_size
        self.J = J
        self.T = T
        self.K = K
        #Save values for performance
        self.exponent_scale = 1 / (K * T)

    def deltaE0(self):
        kernel = torch.tensor([[0., 1, 0], [1, 0, 1], [0, 1, 0]]).view((1, 1, 3, 3))
        return torch.conv2d(self.grid, -self.J * 2 * kernel, padding=1) * self.grid

    # Work-in-Progress
    def subsample(self, size, topleft, bottomright, padding=1):
        #Create grid padded so convolution returns correct dimensions
        dim = (bottomright[0] - topleft[0], bottomright[1] - topleft[1])
        topleft = (topleft[0] - padding, topleft[1] - padding)
        bottomright = (bottomright[0] + padding, bottomright[1] + padding)

        subsample = self.grid[0, 0, topleft[0]:bottomright[0], topleft[1]:bottomright[1]]
        return subsample

    # Work-in-Progress
    def calc_coarse(self):
        sample = self.circle(self.coarse_size, (0,0), (self.max, self.max), 1)
        kernel = torch.tensor([[-1, -1, -1], [-1, 8., -1], [-1, -1, -1]]).view((1, 1, 3, 3))
        sample = torch.conv2d(sample, kernel, stride=1)
        self.coarse = torch.abs(sample)
        return self.coarse

    def step(self):
        dE = self.deltaE0()
        prob = torch.exp(-dE * self.exponent_scale)
        flip = (torch.rand_like(prob) < prob) * (torch.rand_like(self.grid) < 0.1) 
        self.grid = self.grid * (flip * (-2) + 1)
        return -torch.sum(dE)

    def draw(self, window="Ising"):
        img = self.grid.view(self.size_x, -1).numpy()
        ratio = 600/self.size_x
        img = cv2.resize(img, (600, (int)(ratio * self.size_y)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window, img)
        cv2.waitKey(1)

class Ising3D:
    
    def __init__(self, size_x, size_y, size_z, J=1, T=1, K=1, coarse_size=24):
        self.grid = (torch.randint(0, 2, [1, 1, size_x, size_y, size_z]) * 2 - 1).float()
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.J = J
        self.T = T
        self.K = K
        #Save values for performance
        self.exponent_scale = 1 / (K * T)

    # Standard Ising model
    def deltaE0(self): 
        kernel = torch.tensor([[[0., 0, 0], [0, 1, 0], [0, 0, 0]], 
                               [[0., 1, 0], [1, 0, 1], [0, 1, 0]],
                               [[0., 0, 0], [0, 1, 0], [0, 0, 0]]]
                             ).view((1, 1, 3, 3, 3))
        return torch.conv3d(self.grid, -self.J * kernel, padding=1) * self.grid

    # 1/r^2 Energy, calculated for 8 adjacent & diagonal elements
    def deltaE1(self): 
        a = 1/2
        b = 1/3
        kernel = torch.tensor([[[b, a, b], [0, 1, 0], [b, a, b]], 
                               [[a, 1, a], [1, 0, 1], [a, 1, a]],
                               [[b, a, b], [0, 1, 0], [b, a, b]]]
                             ).view((1, 1, 3, 3, 3))
        return torch.conv3d(self.grid, -self.J * kernel, padding=1) * self.grid

    def step(self):
        dE = self.deltaE1()
        prob = torch.exp(-dE * self.exponent_scale) * (torch.rand_like(self.grid) < 0.1) 
        flip = (torch.rand_like(prob) < prob)
        self.grid = self.grid * (flip * (-2) + 1)
        return -torch.sum(dE)

    def draw(self, window="Ising"):
        img = self.grid[0,0,0].view(self.size_y, self.size_z).numpy()
        ratio = 600/self.size_y
        img = cv2.resize(img, (600, (int)(ratio * self.size_z)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window, img)
        cv2.waitKey(1)




# class edge_detect:
    
#     def __init__(self, size):
#         self.size = size
#         self.coarse_size = (int)(size/100)
#         self.grid = (torch.zeros((1, 1, size, size))).float()
#         self.coarse = (torch.zeros((1, 1, self.coarse_size, self.coarse_size))).float()
#         # Some error is contributed from edge effects, padding can fix this
#         # However, this error is 2-3 orders of magnitude smaller than any given precision
#         # of this algorithm. The correction is slow and unnessacary.

#         # self.max = (1. + 1./self.size)
#         self.max = 1

#     def circle(self, size, topleft, bottomright, padding=1):
#         #Create grid padded so convolution returns correct dimensions
#         dim = (bottomright[0] - topleft[0], bottomright[1] - topleft[1])
#         topleft = (topleft[0] - padding * dim[0]/size, topleft[1] - padding * dim[1]/size)
#         bottomright = (bottomright[0] + padding * dim[0]/size, bottomright[1] + padding * dim[1]/size)

#         x = torch.pow(torch.linspace(topleft[0], bottomright[0], size+2*padding), 2)
#         y = torch.pow(torch.linspace(topleft[1], bottomright[1], size+2*padding), 2)
#         return ((x.reshape(-1, 1) + y) < 1).float().view(1, 1, size+2*padding, size+2*padding)

#     def calc_pi(self):
#         total = 0
#         self.grid = torch.abs(self.calc_fine())
#         self.grid = self.grid / self.grid.sum(3, keepdim=True)
#         # removing nan
#         # self.grid[self.grid != self.grid] = 0
#         ind = torch.nonzero(self.grid)
#         factor = (self.max**2)/(self.size**2)
#         print(ind.size(0))
#         for i in range(ind.size(0)):
#             x = ind[i,2]
#             y = ind[i,3]
#             total += self.grid[0,0,x,y] * y
#         return total * factor * 4

#     # Clean this up
#     def calc_fine(self):
#         self.calc_coarse()
#         ind = torch.nonzero(self.coarse)
#         kernel = torch.tensor([[-1, -1, -1], [-1, 8., -1], [-1, -1, -1]]).view((1, 1, 3, 3))
#         factor = self.max/self.size
#         for i in range(ind.size(0)):
#             topleft = [100 * ind[i, 2], 100 * ind[i, 3]]
#             bottomright = [topleft[0] + 100, topleft[1] + 100]
#             sample = self.circle(100, (factor*topleft[0], factor*topleft[1]), (factor*bottomright[0], factor*bottomright[1]), 1)
#             sample = torch.conv2d(sample, kernel, stride=1)
#             self.grid[0, 0, topleft[0]:bottomright[0], topleft[1]:bottomright[1]] = sample
#         return self.grid

#     def calc_coarse(self):
#         sample = self.circle(self.coarse_size, (0,0), (self.max, self.max), 1)
#         kernel = torch.tensor([[-1, -1, -1], [-1, 8., -1], [-1, -1, -1]]).view((1, 1, 3, 3))
#         sample = torch.conv2d(sample, kernel, stride=1)
#         self.coarse = torch.abs(sample)
#         return self.coarse

#     def calc_raw(self):
#         sample = self.circle(self.size, (0,0), (self.max, self.max), 0)
#         total = torch.sum(sample)
#         return 4 * total / (self.size ** 2) 

#     def update(self):
#         self.grid = self.grid + self.calc(0.1)

#     def draw(self, window="circle"):
#         img = self.grid.view(self.grid.size(2), -1).numpy()
#         ratio = 600/self.grid.size(2)
#         img = cv2.resize(img, (600, (int)(ratio * self.grid.size(3))), interpolation=cv2.INTER_NEAREST)
#         cv2.imshow(window, img)
#         cv2.waitKey(0)
        