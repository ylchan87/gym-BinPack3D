import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import copy

from gym_BinPack3D.envs.Container import Container, Box
from gym_BinPack3D.envs.BoxSeqGenerator import BoxSeqGenerator, RandomBoxCreator, CuttingBoxCreator, Rotate


class PackingGame(gym.Env):
    """
    x: depth ( small x = deep inside, large x = near to viewer)
    y: length (small y = left       , large y = right)
    z: height (small z = low        , large z = high)
       Z
       |
       |
       |______Y
      /
     /
    X

    Concept:
    box: you pack boxes into a container
    container: contain boxes
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(   self, 
                    container_size = (20, 20, 20),
                    boxSeqGenerator='random', 
                    enabled_rotations = [Rotate.NOOP],
                    n_foreseeable_box = 1,
                    box_set = [Box(1,1,1), Box(2,3,4)], 
                    minSideLen = None,
                    maxSideLen = None,
                    genValidPlacementMask = True,
                    #data_name = None,  #TODO: load saved box seq
                    **kwags):
        """
        Caveat: order in list "enabled_rotations" affects action meaning.
        below should work, other orders probably not
        [Rotate.NOOP]
        [Rotate.NOOP, Rotate.XY]
        [Rotate.NOOP, Rotate.XY, Rotate.XZ]
        [Rotate.NOOP, Rotate.XY, Rotate.XZ, Rotate.YZ]
        """

        self.container_size = container_size
        self.container_area = int(self.container_size[0] * self.container_size[1])
        self.container_vol  = int(self.container_size[0] * self.container_size[1] * self.container_size[2])
        self.container = Container(*self.container_size)

        self.box_set = box_set
        self.enabled_rotations = enabled_rotations
        self.n_foreseeable_box = n_foreseeable_box

        self.boxSeqGenerator = boxSeqGenerator
        if type(boxSeqGenerator) is str:
            assert box_set is not None
            if boxSeqGenerator == 'random':
                print('using random box sequence')
                self.boxSeqGenerator = RandomBoxCreator(box_set, self.enabled_rotations, n_foreseeable_box)
            elif boxSeqGenerator == 'CUT-1':
                print('using CUT-1 logic box sequence')
                self.boxSeqGenerator = CuttingBoxCreator(container_size, minSideLen, maxSideLen, "ByZ",
                                                         self.enabled_rotations, n_foreseeable_box
                                                         )
            elif boxSeqGenerator == 'CUT-2':
                print('using CUT-2 logic box sequence')
                self.boxSeqGenerator = CuttingBoxCreator(container_size, minSideLen, maxSideLen, "ByStackOrder",
                                                         self.enabled_rotations, n_foreseeable_box,
                                                         )
        assert isinstance(self.boxSeqGenerator, BoxSeqGenerator)    

        self.genValidPlacementMask = genValidPlacementMask

        obsSpace = {
            "height_map"   : gym.spaces.Box(low=0.0, high=self.container_size[2], shape=(self.container_size[0],self.container_size[1]) ),
            "coming_boxes" : gym.spaces.Box(low=0.0, high=max(self.container_size), shape=(self.n_foreseeable_box,3) ),
        }

        if self.genValidPlacementMask:
            obsSpace["valid_placement_mask"] = gym.spaces.MultiBinary( [len(self.enabled_rotations), self.container_size[0],self.container_size[1] ])

        self.action_space = gym.spaces.MultiDiscrete( [self.container_area, len(self.enabled_rotations)] )
        self.observation_space = gym.spaces.Dict(obsSpace)
        

    #def get_box_ratio(self):
    #    coming_box = self.next_box
    #    return (coming_box[0] * coming_box[1] * coming_box[2]) / (self.space.plain_size[0] * self.space.plain_size[1] * self.space.plain_size[2])


    #def get_box_plain(self):
    #    x_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[0]
    #    y_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[1]
    #    z_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[2]
    #    return (x_plain, y_plain, z_plain)
    
    def actionIdx_to_position(self, idx):
        lx = idx // self.container_size[1]
        ly = idx % self.container_size[1]
        return (lx, ly)

    def position_to_actionIdx(self, pos):
        assert len(pos) == 2
        assert pos[0] >= 0 and pos[1] >= 0
        assert pos[0] < self.container_size[0] and pos[1] < self.container_size[1]
        return pos[0] * self.container_size[1] + pos[1]

    @property
    def cur_observation(self):
        hmap = self.container.heightMap
        coming_boxes = self.boxSeqGenerator.next_N_boxes()
        firstBox = coming_boxes[0]
        coming_boxes = np.array([(b.dx,b.dy,b.dz) for b in coming_boxes], dtype=int)

        if self.genValidPlacementMask:
            mask = []
            for r in self.enabled_rotations:
                b = copy.deepcopy(firstBox)
                b.rotate(r)
                mask.append( (self.container.get_possible_positions(b)>0).astype(np.int8) )
            mask = np.array(mask)

        obs =  {
                "height_map"   : hmap,
                "coming_boxes" : coming_boxes
               }
        if self.genValidPlacementMask: obs["valid_placement_mask"] = mask
        
        return obs

    def step(self, action):
        position = self.actionIdx_to_position(action[0])
        rotation = action[1]
        box = self.boxSeqGenerator.next_N_boxes()[0]
        if (rotation == Rotate.XY): box.dx, box.dy = box.dy, box.dx
        if (rotation == Rotate.XZ): box.dx, box.dz = box.dz, box.dx
        if (rotation == Rotate.YZ): box.dy, box.dz = box.dz, box.dy

        succeeded = self.container.drop_box(box, position)

        if succeeded:
            self.boxSeqGenerator.pop_box() # remove current box from the list
            reward = (box.dx*box.dy*box.dz) / self.container_vol * 10
            done = False
        else:            
            reward = 0.0
            done = True
        
        info = {'counter':len(self.container.boxes), 'ratio':self.container.get_fill_ratio()}
        return self.cur_observation, reward, done, info
    
    def reset(self):
        self.boxSeqGenerator.reset()
        self.container.reset()
        return self.cur_observation

    def render(self, mode='human'):
        from gym_BinPack3D.envs.VisUtil import plot_box
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt

        iMode = plt.isinteractive()

        if mode=='human':
            plt.interactive(True)
        if mode=='rgb_array':
            plt.interactive(False)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        maxSideLen = max(self.container_size)
        box = Box(maxSideLen,maxSideLen,maxSideLen)
        plot_box(box, ax, color=(0,0,0,0), showEdges=False) # invisible bound box
        
        box = Box(*self.container_size)
        plot_box(box, ax)
        
        for box in self.container.boxes[:-1]:
            plot_box(box, ax, color=(0,0.5,1,1))
        
        if len(self.container.boxes)>0:
            box = self.container.boxes[-1]
            plot_box(box, ax, color=(0,0.,1,1)) # plot the latest box in diff color

        iMode = plt.interactive(iMode)  #restore
        if mode=='human':
            return fig
        if mode=='rgb_array':
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return image_from_plot

        return None

    def close(self):
        pass

