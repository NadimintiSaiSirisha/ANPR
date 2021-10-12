# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 01:51:09 2021

@author: Siri
"""
import torch.nn.functional as F
import torch
def roi_pooling_ims(input, rois, size=(7, 7), spatial_scale=1.0):
    # written for one roi one image
    # size: (w, h)
    print("type(rois): ", type(rois))
    print("rois: ", rois)
    assert (rois.dim() == 2)
    assert len(input) == len(rois)
    assert (rois.size(1) == 4)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im = input.narrow(0, i, 1)[..., roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
        output.append(F.adaptive_max_pool2d(im, size))

    return torch.cat(output, 0)


h1 = 122
w1 = 122
p1 = torch.FloatTensor([[w1, 0, 0, 0], [0, h1, 0, 0], [0, 0, w1, 0], [0, 0, 0, h1]])
boxNew = torch.FloatTensor([0.4176,0.4040,0.6504,0.4632])
#a = boxNew.mm(p1)
#print(a)
roi1 = roi_pooling_ims(_x1, boxNew.mm(p1), size=(16, 8))
print(roi1)