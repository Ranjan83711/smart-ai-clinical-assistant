import numpy as np

def extract_lines(binary_img):

    projection = np.sum(binary_img == 0, axis=1)

    lines=[]
    start=None

    for i,val in enumerate(projection):
        if val>25 and start is None:
            start=i
        elif val<=25 and start is not None:
            if i-start>18:
                lines.append((start,i))
            start=None

    crops=[]
    for s,e in lines:
        crops.append(binary_img[s:e,:])

    return crops