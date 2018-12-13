import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



def normal_dist(mhi, C, x):
    dim = C.shape[0]

    left = (x - mhi).transpose()
    middle = np.linalg.inv(C)
    right = (x - mhi)

    mahal = np.matmul(np.matmul( (x - mhi).transpose(), np.linalg.inv(C)), x - mhi ) 

    return np.diag(np.exp(-1/2 * mahal ) / np.sqrt( np.power(2.0 * np.pi, dim) * np.linalg.det(C) ))

def random_normal_scatter(mhi, C, num_samples=20):

    A = np.linalg.cholesky(C)

    Z = np.random.normal(size=(C.shape[0],num_samples))

    X = np.matmul(A,Z) + mhi

    return X



if __name__ == '__main__':

    C = np.array([
            [0.5, 0.45, 0.45]
           ,[0.45, 0.5, 0.45]
           ,[0.45, 0.45, 0.5]
        ])

    mhi = np.array([[5.0],[5.0],[5.0] ])

    xs,ys,zs = random_normal_scatter(mhi, C, num_samples=1000)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs,ys,zs)

    # ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')

    std = np.sqrt(np.max(C))

    minmax = (2*std, -2*std)

    ax.set_xlim(minmax + mhi[0])
    ax.set_ylim(minmax + mhi[1]) 
    ax.set_zlim(minmax + mhi[2]) 

    plt.show()

