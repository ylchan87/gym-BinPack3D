import numpy as np
from functools import reduce
import copy, time

from numpy.lib.function_base import percentile

"""
    x: depth  (small x = deep inside, large x = near to viewer)
    y: length (small y = left       , large y = right)
    z: height (small z = low        , large z = high)
       Z
       |
       |
       |______Y
      /
     /
    X
"""

class Box(object):
    def __init__(self, dx, dy, dz, x=0, y=0, z=0):
        """
        dx,dy,dz : size of the box 
        x,y,z    : the position of the deepest-leftmost-lowest corner of the box
        """
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x = x
        self.y = y
        self.z = z

    def standardize(self):
        return tuple([self.dx, self.dy, self.dz, self.x, self.y, self.z])
    
    def __repr__(self):
        return f"Box: Size {self.dx} {self.dy} {self.dz} Position {self.x} {self.y} {self.z}"

class Container(object):
    def __init__(self, dx=10, dy=10, dz=10):
        self.boxes = []
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.heightMap = np.zeros(shape=(dx, dy), dtype=np.int32)

    def reset(self):
        self.boxes = []
        self.heightMap[:,:] = 0

    def regen_height_map(self):
        heightMap = np.zeros_like(self.heightMap)
        for box in self.boxes:
            heightMap = self.update_height_map(heightMap, box)
        return heightMap

    @staticmethod
    def update_height_map(heightMap, box):
        le = box.x
        ri = box.x + box.dx
        up = box.y
        do = box.y + box.dy
        max_h = np.max(heightMap[le:ri, up:do])  #???
        max_h = max(max_h, box.z + box.dz)       #???
        heightMap[le:ri, up:do] = max_h
        return heightMap

    def get_height_map(self):
        return copy.deepcopy(self.heightMap)

    def get_box_list(self):
        return [ box.standardize() for box in self.boxes]

    def get_fill_ratio(self):
        vo = sum([box.dx * box.dy * box.dz for box in self.boxes])
        mx = self.dx * self.dy * self.dz
        ratio = vo / mx
        assert ratio <= 1.0
        return ratio

    def check_box_placement_valid(self, box, pos, checkMode="normal"):
        """
        return -1 if placement is invalid
        return height of the box base when placed, if placement is good

        box: obj of type "Box"
        pos: tuple (x,y)
        checkMode: str "nornal" or "strict"

        at "strict" check the box must be supported 100% below its base
        """
        x, y = pos
        if x+box.dx > self.dx or y+box.dy > self.dy: return -1
        if x < 0 or y < 0: return -1

        rec = self.heightMap[x:x+box.dx, y:y+box.dy]
        r00 = rec[ 0, 0]
        r10 = rec[-1, 0]
        r01 = rec[ 0,-1]
        r11 = rec[-1,-1]
        rm = max(r00,r10,r01,r11)
        supportedCorners = int(r00==rm)+int(r10==rm)+int(r01==rm)+int(r11==rm)
        if supportedCorners < 3: return -1

        max_h = np.max(rec)
        assert max_h >= 0
        if max_h + box.dz > self.dz: return -1
        
        # check box base is well supported
        max_area = np.sum(rec==max_h)
        area = box.dx * box.dy

        if checkMode == "strict" and max_area<area: return -1

        if max_area/area > 0.95: 
            return max_h
        if rm == max_h and supportedCorners == 3 and max_area/area > 0.85:
            return max_h
        if rm == max_h and supportedCorners == 4 and max_area/area > 0.50:
            return max_h

        return -1

    def get_possible_positions(self, box):
        """
        find possible position to place the incoming box, 
        the position shoulf satisfy stability and be accessable
        """
        action_mask = np.zeros(shape=(self.dx, self.dy), dtype=np.int32)
        
        for i in range(self.dx-box.dx+1):
            for j in range(self.dy-box.dy+1):
                pos = (i,j)
                if self.check_box_placement_valid(box, pos) >= 0:
                    action_mask[i, j] = 1

        return action_mask

    def drop_box(self, box, pos):
        """
        place a box at pos into the container
        """
        x, y = pos
        new_h = self.check_box_placement_valid(box, pos)
        if new_h == -1: return False

        box.x, box.y, box.z = x, y, new_h
        self.boxes.append(copy.deepcopy(box))
        self.heightMap = self.update_height_map(self.heightMap, box)
        return True

    @staticmethod
    def pretty_print_bool_2D_array(arr):
        """
        (0,0) at upper left corner of screen 
        down is +x
        right is +y
        """
        dx,dy = arr.shape
        for x in range(dx):
            print ("".join( ["O" if b else "X" for b in arr[x] ]) )

if __name__=="__main__":
    print("Case 1: normal box place")
    container = Container(5,8,10)
    box1 = Box(4,4,5)
    mask = container.get_possible_positions(box1)
    Container.pretty_print_bool_2D_array(mask)
    print("====")

    print("Case 2: long box")
    box1 = Box(12,4,6)
    mask = container.get_possible_positions(box1)
    Container.pretty_print_bool_2D_array(mask)
    print("====")

    print("Case 3: small box")
    box1 = Box(1,1,1)
    mask = container.get_possible_positions(box1)
    Container.pretty_print_bool_2D_array(mask)
    print("====")

    print("Case 4: after small box placed")
    box1 = Box(4,4,5)
    box2 = Box(1,1,1)
    container.drop_box(box2, (2,4) )
    mask = container.get_possible_positions(box1)
    Container.pretty_print_bool_2D_array(mask)
    print("====")

    print("Case 5: reset and place normal box")
    box1 = Box(4,4,5)
    container.reset()
    mask = container.get_possible_positions(box1)
    Container.pretty_print_bool_2D_array(mask)
    print("Place succeed?", container.drop_box(box1, (2,4) ) )
    print("Place succeed?", container.drop_box(box1, (1,4) ) )
    print(container.boxes)
    print("====")

    print("Case 6: stack up box")
    box1 = Box(4,4,5)
    box2 = Box(3,3,5)
    box3 = Box(1,1,5)
    container.reset()
    print("...........")
    mask = container.get_possible_positions(box1)
    Container.pretty_print_bool_2D_array(mask)
    print("Place succeed?", container.drop_box(box1, (1,4) ) )

    print("...........")
    mask = container.get_possible_positions(box2)
    Container.pretty_print_bool_2D_array(mask)
    print("Place succeed?", container.drop_box(box2, (1,4) ) )

    print("...........")
    mask = container.get_possible_positions(box3)
    Container.pretty_print_bool_2D_array(mask)
    print("Place succeed?", container.drop_box(box3, (0,0) ) )
    
    print("...........")
    print(container.boxes)
    print("====")
        


