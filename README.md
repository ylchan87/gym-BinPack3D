## About

This is an OpenAI gym environment for the 3D Bin Packing game.

The game is like "3D tetris": to pack coming boxes to a container as space efficient as possible.

- You see 1 or more up coming boxes.
- You choose a x,y position and rotation of the box
- You place the box into the container and "score"
- The space to place the box should be accesable from the top, and the box should be physically stable after the placement

This "game" is commonly played in logistic, eg packing trucks with items from conveyer belt.

Code refactored from https://github.com/alexfrom0815/Online-3D-BPP-DRL

## Reference
https://github.com/alexfrom0815/Online-3D-BPP-DRL

https://github.com/openai/gym/blob/master/docs/creating-environments.md


## Install
```
#after clone
cd gym-BinPack3D
pip install -e .
```

## Doc

See MinimalExample.ipynb for usage

