import numpy as np
import matplotlib.pyplot as plt
from typing import List

def sample_no_signalling(num: int=1, local=None, output_format=0)->List[np.array]:
    sample = [sample_no_signalling_one(local) for _ in range(num)]
    if output_format == 0:
        post_process = lambda x: x.flatten()
    elif output_format == 1:
        post_process = generate_img
    elif output_format == 2:
        post_process = generate_img_gradient
    else:
        print('Not valid output format')
        return
    return list(map(post_process, sample))


def sample_no_signalling_one(local=None)->np.array:
    A, B, C = np.random.uniform(size=(3,2,2))
    A = A / np.sum(A)
    B[0,:] = np.sum(A[0,:]) * B[0,:] / np.sum(B[0,:])
    B[1,:] = np.sum(A[1,:]) * B[1,:] / np.sum(B[1,:])
    C[:,0] = np.sum(B[:,0]) * C[:,0] / np.sum(C[:,0])
    C[:,1] = np.sum(B[:,1]) * C[:,1] / np.sum(C[:,1])
    min_sum = np.min(np.minimum(np.sum(A, axis=1), np.sum(C, axis=0)))
    d1 = np.random.uniform(0, min_sum)
    d2 = np.sum(C, axis=1)[0] - d1
    d3 = np.sum(A, axis=0)[0] - d1
    d4 = np.sum(A, axis=0)[1] - d2
    D = np.array([[d1, d2], [d3, d4]])
    result = np.concatenate((np.concatenate((A, B), axis=1), np.concatenate((D, C),axis=1)), axis=0)
    if local is None:
        if check_correctness(result):
            return result
        else:
            return sample_no_signalling_one()
    else:
        if local==check_locality(result) and check_correctness(result):
            return result
        else:
            return sample_no_signalling_one(local)
        
def check_correctness(box: np.array)->bool:
    row_check = np.allclose(np.sum(box[:,:2], axis=1), np.sum(box[:,2:], axis=1))
    col_check = np.allclose(np.sum(box[:2,:], axis=0), np.sum(box[2:,:], axis=0))
    return np.all(row_check) and np.all(col_check) and np.all(box.flatten()>=0)

def generate_img(box: np.array, resolution = 100)->np.array:
    img = []
    for row in box:
        img_row = []
        for element in row:
            img_tmp = np.ones(shape=(resolution, resolution))
            img_tmp[:, :int(resolution*element)] = 0
            img_row.append(img_tmp)
        img.append(np.hstack(img_row))
    img = np.vstack(img)
    return img

def generate_img_gradient(box: np.array, resolution: int=8)->np.array:
    img = []
    for row in box:
        img.append(np.hstack([np.ones(shape=(resolution, resolution)) * element for element in row]))
    return np.vstack(img)

def check_locality(box: np.array)->bool:
    p00_00, p10_00, p00_10, p10_10 = box[0,:]
    p01_00, p11_00, p01_10, p11_10 = box[1,:]
    p00_01, p10_01, p00_11, p10_11 = box[2,:]
    p01_01, p11_01, p01_11, p11_11 = box[3,:]
    
    c1 = p01_00 + p10_00 + p00_10 + p11_10 + p00_01 + p11_01 + p00_11 + p11_11
    c2 = p00_00 + p11_00 + p01_10 + p10_10 + p00_01 + p11_01 + p00_11 + p11_11
    c3 = p00_00 + p11_00 + p00_10 + p11_10 + p01_01 + p10_01 + p00_11 + p11_11
    c4 = p00_00 + p11_00 + p00_10 + p11_10 + p00_01 + p11_01 + p01_11 + p10_11
    
    c5 = p11_00 + p00_00 + p01_10 + p10_10 + p01_01 + p10_01 + p01_11 + p10_11
    c6 = p01_00 + p10_00 + p11_10 + p00_10 + p01_01 + p10_01 + p01_11 + p10_11
    c7 = p01_00 + p10_00 + p01_10 + p10_10 + p11_01 + p00_01 + p01_11 + p10_11
    c8 = p01_00 + p10_00 + p01_10 + p10_10 + p01_01 + p10_01 + p11_11 + p00_11
    
    return np.all([inq < 3 for inq in [c1, c2, c3, c4, c5, c6, c7, c8]])