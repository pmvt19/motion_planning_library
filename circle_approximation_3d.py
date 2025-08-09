import math
import numpy as np
import matplotlib.pyplot as plt

def rect_prism_to_circles_x_short(aa_rect_prism):
    # aa_rect (x,y,z,xl,yl,zl)

    radii = aa_rect_prism[3] / 2
    x, y, z, xl, yl, zl = aa_rect_prism

    segment_lengths = yl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    ys = [y-yl/2+radii]
    for i in range(num_points):
        ys.append(ys[-1] + delta)

    segment_lengths = zl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    zs = [z-zl/2+radii]
    for i in range(num_points):
        zs.append(zs[-1] + delta)

    ys = np.array(ys)
    zs = np.array(zs)

    output = np.array(np.meshgrid(ys,zs)).T.reshape(-1, 2)
    points = np.hstack((np.ones((output.shape[0],1))*x, output, np.ones((output.shape[0],1))*xl/2))

    return points

def rect_prism_to_circles_y_short(aa_rect_prism):
    # aa_rect (x,y,z,xl,yl,zl)

    radii = aa_rect_prism[4] / 2
    x, y, z, xl, yl, zl = aa_rect_prism

    segment_lengths = xl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    xs = [x-xl/2+radii]
    for i in range(num_points):
        xs.append(xs[-1] + delta)

    segment_lengths = zl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    zs = [z-zl/2+radii]
    for i in range(num_points):
        zs.append(zs[-1] + delta)

    xs = np.array(xs)
    zs = np.array(zs)

    output = np.array(np.meshgrid(xs,zs)).T.reshape(-1, 2)
    points = np.hstack((output[:, 1].reshape(-1, 1), np.ones((output.shape[0],1))*y, output[:, 0].reshape(-1, 1), np.ones((output.shape[0],1))*yl/2))

    return points

def rect_prism_to_circles_z_short(aa_rect_prism):
    # aa_rect (x,y,z,xl,yl,zl)

    radii = aa_rect_prism[5] / 2
    x, y, z, xl, yl, zl = aa_rect_prism

    segment_lengths = xl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    xs = [x-xl/2+radii]
    for i in range(num_points):
        xs.append(xs[-1] + delta)

    segment_lengths = yl - (2*radii)
    num_points = math.ceil(segment_lengths / (2*radii))
    delta = segment_lengths / num_points

    ys = [y-yl/2+radii]
    for i in range(num_points):
        ys.append(ys[-1] + delta)

    xs = np.array(xs)
    ys = np.array(ys)

    output = np.array(np.meshgrid(xs,ys)).T.reshape(-1, 2)
    points = np.hstack((output, np.ones((output.shape[0],1))*z, np.ones((output.shape[0],1))*zl/2))

    return points

def rect_prisms_to_circles(aa_rect_prisms):
    """
    aa_rect_prisms: (B, 6)
    """

    dim_sizes = aa_rect_prisms[:, 3:6]
    min_dims = np.argmin(dim_sizes, axis=1)

    x_min_dim_mask = min_dims[min_dims==0]
    y_min_dim_mask = min_dims[min_dims==1]
    z_min_dim_mask = min_dims[min_dims==2]

    x_min_prisms = aa_rect_prisms[x_min_dim_mask]
    y_min_prisms = aa_rect_prisms[y_min_dim_mask]
    z_min_prisms = aa_rect_prisms[z_min_dim_mask]

    circles = []
    for prism in x_min_prisms:
        circles.append(rect_prism_to_circles_x_short(prism))
    for prism in y_min_prisms:
        circles.append(rect_prism_to_circles_y_short(prism))
    for prism in z_min_prisms:
        circles.append(rect_prism_to_circles_z_short(prism))
    return np.array(circles)
        

    # rect_prism_to_circles_x_short(aa_rect_prisms[x_min_dim_mask])
    # rect_prism_to_circles_y_short(aa_rect_prisms[y_min_dim_mask])
    # rect_prism_to_circles_z_short(aa_rect_prisms[z_min_dim_mask])

def xyzwhl_to_ordered_vertices(aa_rect_prism):
    # aa_rect (x,y,z,xl,yl,zl)
    x, y, z, xl, yl, zl = aa_rect_prism
    vertices = np.array([
        [x-xl/2, y-yl/2, z-zl/2],
        [x+xl/2, y-yl/2, z-zl/2],
        [x+xl/2, y+yl/2, z-zl/2],
        [x-xl/2, y+yl/2, z-zl/2],
        [x-xl/2, y-yl/2, z+zl/2],
        [x+xl/2, y-yl/2, z+zl/2],
        [x+xl/2, y+yl/2, z+zl/2],
        [x-xl/2, y+yl/2, z+zl/2],
    ])
    return vertices

def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

if __name__ == '__main__':

    center = (0,0,0)
    lengths = (2,2,2)

    edges = np.array([
        [0,1], # Bottom Face
        [1,2],
        [2,3],
        [3,0], # Bottom Face
        
        [4,5], # Top Face
        [5,6],
        [6,7],
        [7,4], # Top Face

        [0,4], # Mid Faces
        [1,5],
        [2,6],
        [3,7], # Mid Faces
    ])

    # vertices = np.array([
    #     [1,1,-1], # 0
    #     [1,-1,-1], # 1
    #     [-1,-1,-1], # 2
    #     [-1,1,-1], # 3 
    #     [1,1,1], # 4 
    #     [1,-1,1], # 5
    #     [-1,-1,1], # 6 
    #     [-1,1,1], # 7
    # ])

    # ax = plt.axes(projection='3d')
    # for a,b in edges:
    #     point_a = vertices[a]
    #     point_b = vertices[b]

    #     ax.plot3D([point_a[0], point_b[0]],[point_a[1], point_b[1]],[point_a[2], point_b[2]])
    # plt.show()

    aa_rect_prism1 = np.array([0,0,0,1,5,5])
    aa_rect_prism2 = np.array([0,2.5,0,5,1,5])
    aa_rect_prism3 = np.array([0,-2.5,0,5,1,5])

    prisms = np.array([aa_rect_prism1, aa_rect_prism3])
    circles = rect_prisms_to_circles(prisms).reshape(-1, 4)

    ax = plt.axes(projection='3d')
    for prism in prisms:
        ordered_verts = xyzwhl_to_ordered_vertices(prism)
        for a,b in edges:
            point_a = ordered_verts[a]
            point_b = ordered_verts[b]
            ax.plot3D([point_a[0], point_b[0]],[point_a[1], point_b[1]],[point_a[2], point_b[2]])
        ax.scatter(circles[:, 0], circles[:, 1], circles[:, 2])
        # ax.set_box_aspect([[-5,5],[-5,5],[-5,5]])
        ax.set_box_aspect([1,1,1])
        amin = -11
        amax = 11
        ax.set_xlim(amin, amax)
        ax.set_ylim(amin, amax)
        ax.set_zlim(amin, amax)
    
    for x,y,z,r in circles:
        # cpx, cpy, cpz = drawSphere(x,y,z,1*math.sqrt(3))
        cpx, cpy, cpz = drawSphere(x,y,z,r)
        # ax.plot_wireframe(cpx, cpy, cpz, color="r")
        ax.plot_surface(cpx, cpy, cpz, color="r")
    plt.show()


    # aa_rect_prism = np.array([0,0,0,2,3,3.1])
    # aa_rect_prism = np.array([0,0,0,2,3.1,3])

    # aa_rect_prism = np.array([0,0,0,2,11.34,21.67])


    # aa_rect_prism = np.array([0,0,0,20.82,2,21.67])


    # aa_rect_prism = np.array([0,0,0,20.82,21.67,2])
    # ordered_verts = xyzwhl_to_ordered_vertices(aa_rect_prism)

    # ax = plt.axes(projection='3d')
    # for a,b in edges:
    #     point_a = ordered_verts[a]
    #     point_b = ordered_verts[b]
    #     ax.plot3D([point_a[0], point_b[0]],[point_a[1], point_b[1]],[point_a[2], point_b[2]])
    # ax.scatter(circles[:, 0], circles[:, 1], circles[:, 2])
    # # ax.set_box_aspect([[-5,5],[-5,5],[-5,5]])
    # ax.set_box_aspect([1,1,1])
    # amin = -11
    # amax = 11
    # ax.set_xlim(amin, amax)
    # ax.set_ylim(amin, amax)
    # ax.set_zlim(amin, amax)

    # for x,y,z,r in circles:
    #     # cpx, cpy, cpz = drawSphere(x,y,z,1*math.sqrt(3))
    #     cpx, cpy, cpz = drawSphere(x,y,z,r)
    #     # ax.plot_wireframe(cpx, cpy, cpz, color="r")
    #     ax.plot_surface(cpx, cpy, cpz, color="r")

    # plt.show()


