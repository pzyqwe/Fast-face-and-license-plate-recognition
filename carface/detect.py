# -*- coding: UTF-8 -*-
import torch
import copy
from models.experimental import attempt_load
from util.general import check_img_size, non_max_suppression_face, scale_coords
from util.face import *
from time import time
import os
def init(weights, device):
    model = attempt_load(weights, map_location=device)
    return model

def detect_one(model, orgimg, device,file):
    img_size =960
    conf_thres = 0.3
    iou_thres = 0.4
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
    imgsz = check_img_size(img_size, s=model.stride.max())
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()
            print('facenum:',len(det))
            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xyxy, conf, landmarks, class_num)
    cv2.imwrite('output/'+file, orgimg)




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init('carface.pt', device)
    filedir=os.listdir('images/')
    for file in filedir:
        orgimg = cv2.imread('images/'+file)
        start=time()
        detect_one(model, orgimg, device,file)
        end=time()
        print(end-start)
