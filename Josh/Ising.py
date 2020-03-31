import torch
import cv2
import numpy as np

# Not a complete program
# For now, it defines a lattice of dipole spins, displays it, and finds
# the Hamiltonian of the lattice.

def energy(lattice,side,J):
    sum = 0
    # 2 adjacent neighbors if in corner, 3 if on 1 edge, 4 otherwise
    # I'm assuming a finite number of dipoles in a region
    for i in range(side):
        for j in range(side):
            if i == 0:
                if j == 0:
                    cont1 = lattice[i][j] * lattice[i][j + 1]
                    cont2 = lattice[i][j] * lattice[i + 1][j]
                    sum += (cont1 + cont2)
                elif j == (side - 1):
                    cont1 = lattice[i][j] * lattice[i][j - 1]
                    cont2 = lattice[i][j] * lattice[i + 1][j]
                    sum += (cont1 + cont2)
                else:
                    cont1 = lattice[i][j] * lattice[i][j - 1]
                    cont2 = lattice[i][j] * lattice[i + 1][j]
                    cont3 = lattice[i][j] * lattice[i][j + 1]
                    sum += (cont1 + cont2 + cont3)
            elif i == (side - 1):
                if j == 0:
                    cont1 = lattice[i][j] * lattice[i][j + 1]
                    cont2 = lattice[i][j] * lattice[i - 1][j]
                    sum += (cont1 + cont2)
                elif j == (side - 1):
                    cont1 = lattice[i][j] * lattice[i][j - 1]
                    cont2 = lattice[i][j] * lattice[i - 1][j]
                    sum += (cont1 + cont2)
                else:
                    cont1 = lattice[i][j] * lattice[i][j - 1]
                    cont2 = lattice[i][j] * lattice[i - 1][j]
                    cont3 = lattice[i][j] * lattice[i][j + 1]
                    sum += (cont1 + cont2 + cont3)
            elif j == 0:
                cont1 = lattice[i][j] * lattice[i + 1][j]
                cont2 = lattice[i][j] * lattice[i - 1][j]
                cont3 = lattice[i][j] * lattice[i][j + 1]
                sum += (cont1 + cont2 + cont3)
            elif j == (side - 1):
                cont1 = lattice[i][j] * lattice[i + 1][j]
                cont2 = lattice[i][j] * lattice[i - 1][j]
                cont3 = lattice[i][j] * lattice[i][j - 1]
                sum += (cont1 + cont2 + cont3)
            else:
                cont1 = lattice[i][j] * lattice[i][j - 1]
                cont2 = lattice[i][j] * lattice[i - 1][j]
                cont3 = lattice[i][j] * lattice[i][j + 1]
                cont4 = lattice[i][j] * lattice[i + 1][j]
                sum += (cont1 + cont2 + cont3 + cont4)
    H = -J*sum
    return H.item()

def draw(tensor, side):
    img = np.zeros((side, side, 3), np.uint8)
    for i in range(side):
        for j in range(side):
            if tensor[i][j] == -1:
                img[i][j] = 0
            else:
                img[i][j] = 255
    img = cv2.resize(img, (3*side, 3*side))
    cv2.imshow("Ising Model", img)
    cv2.waitKey(1)
    input("Enter any key: ")

def main():
    side = 200
    J = 1
    # Note: The max/min Hamiltonian is quadratic wrt the lattice's side length
    maxenergy = -J * (-4 * (side - 2)**2 - 12 * (side - 2) - 8)
    minenergy = -1 * maxenergy
    lattice = 2*torch.randint(0, 2, [side, side])-1
    # Because of how the lattice is sampled, the initial Hamiltonian tends to avg around 0

    draw(lattice, side)
    #print(lattice)
    print("Hamiltonian:", energy(lattice, side, J))
    print("Max Energy:", maxenergy, "\nMin Energy:", minenergy)

main()
