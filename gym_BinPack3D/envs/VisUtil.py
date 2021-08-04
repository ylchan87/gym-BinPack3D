import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

from gym_BinPack3D.envs.Container import Box

def plot_parallelepiped(cube_definition, ax, color=None, showEdges=True):
    """
    Draw a 3D parallelepiped to a matplotlib 3d plot
    
    
    cube_definition: corner, plus 3 pts around that corner eg.
            [(0,0,0), (0,1,0), (1,0,0), (0,0,0.1)]
            
    ax: a matplotlib 3d axis obj i.e. from:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
    modified from: https://stackoverflow.com/questions/44881885/python-draw-parallelepiped
    """
    if color is None: color = (0,0,1,0.1)
        
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    
    #ax = fig.add_subplot(111, projection='3d')
    edgecolors = 'k' if showEdges else (0,0,0,0)
    faces = Poly3DCollection(edges, linewidths=1, edgecolors=edgecolors)
    faces.set_facecolor(color)

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    ax.set_aspect('auto')


def plot_box(box, ax, color=None, showEdges=True):
    """
    box : obj of type "Box"
    
    ax: a matplotlib 3d axis obj i.e. from:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
    """
    dx, dy, dz = box.dx, box.dy, box.dz
    x,y,z = box.x, box.y, box.z
    
    cube_definition = [(x,y,z), 
                       (x+dx,y,z),
                       (x,y+dy,z),
                       (x,y,z+dz),
                      ]
    #print (cube_definition)
    plot_parallelepiped(cube_definition, ax, color, showEdges)