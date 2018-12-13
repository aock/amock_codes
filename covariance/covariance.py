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

    # x = np.array([[2.0],[0.0]])

    # X = normal_dist(mhi, C, x)
    

    # x = np.linspace(mhi[0] - C[0,0]*3.0, mhi[0] + C[0,0]*3.0, num_samples)
    # y = np.linspace(mhi[1] - C[1,1]*3.0, mhi[1] + C[1,1]*3.0, num_samples)


    x = np.random.normal(mhi[0], np.sqrt(C[0,0])*2.0, (num_samples,) )
    y = np.random.normal(mhi[1], np.sqrt(C[1,1])*2.0, (num_samples,) )

    x = np.sort(x)
    y = np.sort(y)

    xx, yy = np.meshgrid(x, y)

    xx = xx.flatten()
    yy = yy.flatten()
    # print(xx.shape)
    # print(yy.shape)

    data = np.array([xx, yy])

    # print(data.shape)

    Z = normal_dist(mhi, C, data)

    xs = np.reshape(xx, (num_samples,num_samples))
    ys = np.reshape(yy, (num_samples,num_samples))
    zs = np.reshape(Z, (num_samples,num_samples))

    return xs, ys, zs



if __name__ == '__main__':

    C = np.array([
            [10.0, 9.0]
           ,[9.0, 10.0]
        ])

    mhi = np.array([[5.0],[5.0] ])

    xs,ys,zs = random_normal_scatter(mhi, C, num_samples=50)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm, linewidth=0, antialiased=False)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability')

    plt.show()

