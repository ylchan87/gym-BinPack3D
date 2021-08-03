import numpy as np
import copy
from Container import Box, Container

from enum import Enum

#FIXME proper seed for rand nums

class Rotate(Enum):
   NOOP = 0  # a.k.a. No operation
   XY  = 1
   XZ  = 2
   YZ  = 3

class BoxSeqGenerator(object):
    def __init__(self, enabled_rotations = None, n_foreseeable_box = None, seed=None):
        """
        enabled_rotations = list of Enum Rotate
        n_foreseeable_box = int>=1
        """
        if enabled_rotations is None: enabled_rotations = [Rotate.NOOP]
        if n_foreseeable_box is None: n_foreseeable_box = 1

        self.box_list = []
        self.enabled_rotations = enabled_rotations
        self.n_foreseeable_box = n_foreseeable_box

        self.seed = seed
        if seed is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.reset()

    def reset(self):
        self.box_list.clear()
        self._gen_more_boxes()

    def next_N_boxes(self):
        return self.box_list[:self.n_foreseeable_box]

    def pop_box(self, idx = 0):
        assert len(self.box_list) > idx
        assert idx < self.n_foreseeable_box
        self.box_list.pop(idx)
        self._gen_more_boxes()
    
    def _rotate_box(self, box):
        rotation = self.rng.choice(self.enabled_rotations)
        if (rotation == Rotate.XY): box.dx, box.dy = box.dy, box.dx
        if (rotation == Rotate.XZ): box.dx, box.dz = box.dz, box.dx
        if (rotation == Rotate.YZ): box.dy, box.dz = box.dz, box.dy
        return box

    def _gen_more_boxes(self):
        raise NotImplementedError

class RandomBoxCreator(BoxSeqGenerator):
    """
    Random gen box from a given list
    
    enabled_rotations : list of Enum Rotate
    n_foreseeable_box : int>=1
    box_set : list of obj of type "Box"    
    """
    default_box_set = [ Box(1,1,1), Box(1,3,5)]

    def __init__(self, box_set=None, *args, **kw):
        if box_set is None: box_set = RandomBoxCreator.default_box_set
        self.box_set = box_set
        super().__init__(*args, **kw)

        print ("Box to be sampled:")
        for b in self.box_set: print (b)

    def _gen_more_boxes(self):
        while len(self.box_list)<self.n_foreseeable_box:
            newBox = copy.deepcopy(self.rng.choice(self.box_set))
            newBox = self._rotate_box(newBox)
            self.box_list.append( newBox)        

class CuttingBoxCreator(BoxSeqGenerator):
    """
    Recursively bisect a container (i.e. a big box) to small boxes
    Thus the box sequence generated is guaranteed to have a perfect way to pack into the container

    enabled_rotations : list of Enum Rotate
    n_foreseeable_box : int>=1
    container_zise    : tuple (x,y,z)
    minSideLen : int
    maxSideLen : int
    sortMethod : str, "ByZ" or "ByStackOrder"

    will cut until minSideLen <= box side <= maxSideLen, for all 3 side of the boxes after but    
    """
    def __init__(self, container_size, minSideLen = None, maxSideLen = None, sortMethod = "ByZ", *args, **kw):
        if minSideLen is None: minSideLen = max(1, int(min(container_size)/5) )
        if maxSideLen is None: maxSideLen = max(1, int(min(container_size)/2) )

        self.container_size = container_size
        self.minSideLen = minSideLen
        self.maxSideLen = maxSideLen
        self.sortMethod = sortMethod

        assert minSideLen*2 < max(container_size)
        assert maxSideLen < min(container_size)
        assert sortMethod in ["ByZ", "ByStackOrder"]

        super().__init__(*args, **kw)

    def _gen_more_boxes(self):
        if len(self.box_list)>=self.n_foreseeable_box: return

        # we wipe out old boxes, so no mix of boxes from cut last time 
        self.box_list.clear() 

        # gen new box seq
        rawBox = Box(*self.container_size)
        self._cut_box(rawBox)
        self._sort_boxes()
        self.box_list = [self._rotate_box(b) for b in self.box_list]

        # ensure have some dummy box for observer to see even when all boxes are packed
        for i in range(self.n_foreseeable_box): self.box_list.append( Box(1,1,1) ) 


    def _check_box_size_valid(self, box):
        return  ( (self.minSideLen <= box.dx <= self.maxSideLen) and
                  (self.minSideLen <= box.dy <= self.maxSideLen) and
                  (self.minSideLen <= box.dz <= self.maxSideLen)
                )

    def _cut_box(self, box):
        # enum
        splitX = 0
        splitY = 1
        splitZ = 2
        possibleSplitActions = []
        if box.dx >= self.minSideLen*2 and box.dx > self.maxSideLen: possibleSplitActions.append(splitX)
        if box.dy >= self.minSideLen*2 and box.dy > self.maxSideLen: possibleSplitActions.append(splitY)
        if box.dz >= self.minSideLen*2 and box.dz > self.maxSideLen: possibleSplitActions.append(splitZ)

        if len(possibleSplitActions)==0:
            assert self._check_box_size_valid(box)
            self.box_list.append(box)
            return

        action = self.rng.choice(possibleSplitActions)
        if (action==splitX): pos_range = (self.minSideLen, box.dx - self.minSideLen + 1)
        if (action==splitY): pos_range = (self.minSideLen, box.dy - self.minSideLen + 1)
        if (action==splitZ): pos_range = (self.minSideLen, box.dz - self.minSideLen + 1)
        splitPos = self.rng.integers( pos_range[0], pos_range[1] )

        boxA = copy.deepcopy(box)
        boxB = copy.deepcopy(box)

        if (action==splitX):
            boxA.dx, boxB.dx = splitPos, box.dx-splitPos
            boxB.x += splitPos
        elif (action==splitY):
            boxA.dy, boxB.dy = splitPos, box.dy-splitPos
            boxB.y += splitPos
        elif (action==splitZ):
            boxA.dz, boxB.dz = splitPos, box.dz-splitPos
            boxB.z += splitPos
        else:
            print(f"Unknow action {action}")
            raise ValueError

        self._cut_box(boxA)
        self._cut_box(boxB)

    def _sort_boxes(self):
        self.rng.shuffle(self.box_list) # to break the neat arrangement due to our cut method

        if self.sortMethod == "ByZ":    
            self.box_list = sorted(self.box_list, key = lambda b: b.z)
        elif self.sortMethod == "ByStackOrder":
            reordered_box_list = []
            container = Container(*self.container_size)

            idx = 0
            placedOnce = False
            while len(self.box_list)>0:
                box = self.box_list[idx]
                pos = (box.x,box.y) # the box's x,y,z is set when it was cut and hint the correct way to place box
                canPlaceHeight = container.check_box_placement_valid(box, pos, checkMode="strict")
                if canPlaceHeight>=0 and canPlaceHeight==box.z:
                    box = self.box_list.pop(idx)
                    reordered_box_list.append(box)
                    container.drop_box(box, pos)
                    placedOnce = True
                else:
                    idx+=1
                
                if idx>=len(self.box_list):
                    if not placedOnce: 
                        print ("All boxes left cannot be placed")
                        raise ValueError
                        break
                    idx=0
                    placedOnce = False
            assert container.get_fill_ratio() == 1.0
            self.box_list = reordered_box_list

if __name__=="__main__":
    boxSeqGenerator = CuttingBoxCreator( (7,9,11), minSideLen = 2, maxSideLen = 5, n_foreseeable_box = 3, sortMethod = "ByStackOrder")