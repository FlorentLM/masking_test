# "Layer-based" salient object detection

Testing a simple but fast masking technique for isolating the subject in captures made with [scAnt](https://github.com/evo-biomech/scAnt)

## Approach

1. Convert the source image into HSV and LAB colour spaces
2. Detect which areas are in focus using Sobel filters on the S and L channels independently
3. Use the H, S, L, A and B channels to isolate "layers" with dark areas, coloured areas, high light areas, etc.
4. Clip the upper or lower bounds in each layer to get rid of unwanted noise/ghosting effects
4. Merge it all back and binarise