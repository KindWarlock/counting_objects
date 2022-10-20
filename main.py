import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.morphology
from skimage.measure import label


arr = np.array([[0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,1,1,1,1,0,0],
                [0,0,0,0,1,1,1,1,0,0],
                [0,0,0,1,1,1,1,1,0,0],
                [0,0,0,0,1,1,1,1,0,0],
                [0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]])

struct = np.ones((5,5))
struct_x = np.zeros((5,5))
for i in range (struct_x.shape[0]):
    struct_x[i, i] = 1
struct_x[0, 4] = 1
struct_x[1, 3] = 1
struct_x[3, 1] = 1
struct_x[4, 0] = 1

struct_plus = np.zeros((5,5))
struct_plus[:,2] = 1
struct_plus[2, :] = 1

def translation(image, vector):
    translated = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            ny = y + vector[0]
            nx = x + vector[1]
            if ny < 0 or nx < 0:
                continue
            if ny >= image.shape[0] or nx >= image.shape[1]:
                continue
            translated[ny, nx] = image[y, x]
    return translated


def dilation(arr):
    result = np.zeros_like(arr)
    for y in range(1, arr.shape[0] - 1):
        for x in range(1, arr.shape[1] - 1):
            rlog = np.logical_and(arr[y,x], struct)
            result[y-1:y+2, x-1:x+2] = np.logical_or(result[y-1:y+2, x-1:x+2], rlog)
    return result


def erosion(arr):
    result = np.zeros_like(arr)
    for y in range(1, arr.shape[0] - 1):
        for x in range(1, arr.shape[1] - 1):
            sub = arr[y-2:y+3, x-2:x+3]
            pos = struct > 0
            if np.all(sub[pos] == struct[pos]):
                result[y, x] = 1
    return result

def closing(arr):
    return erosion(dilation(arr))

def opening(arr):
    return dilation(erosion(arr))


arr = np.load('ps.npy.txt')
labels = ['[]', ']', '[', 'u', 'n']

base_mask = np.ones((4, 6))
left_mask = np.ones((6, 4))
left_mask[2:4, :2] = 0
right_mask = np.ones((6, 4))
right_mask[2:4, 2:] = 0
up_mask = np.ones((4, 6))
up_mask[:2, 2:4] = 0
down_mask = np.ones((4, 6))
down_mask[2:, 2:4] = 0

masks = {'[]': base_mask, ']': left_mask, '[': right_mask, 'u': up_mask, 'n': down_mask}
base_num = 0
sum = 0

for i in labels:
    masked = scipy.ndimage.binary_erosion(arr, structure=masks[i])
    if i == ']' or i == '[':
        labeled = label(masked).max()
    elif i == '[]':
        base_num = label(masked).max()
        labeled = base_num
    else:
        labeled = label(masked).max() - base_num
    sum += labeled
    print(f'{i}: {labeled}')

print(f'All: {sum}')
#
# OUTPUT 
#
# []: 92
# ]: 123
# [: 94
# u: 95
# n: 96
# All: 500