from plate_recognition.plateNet import myNet_ocr
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import sys


def cv_imread(path):  # 可以读取中文路径的图片
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


def allFilePath(rootPath, allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            if temp.endswith('.jpg') or temp.endswith('.png') or temp.endswith('.JPG'):
                allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
plate_Name=np.array(['#','京','沪','津','渝','冀','晋','蒙','辽','吉','黑','苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤','桂','琼',
                             '川','贵','云','藏','陕','甘','青','宁','新','学','警','港','澳','挂','使','领','民','航','深','0','1',
                             '2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R',
                             'S','T','U','V','W','X','Y','Z'])
#plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航深0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
mean_value, std_value = (0.588, 0.193)


def decodePlate(preds):
    pre = 0
    newPreds = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]
    return newPreds


def image_processing(images, device):
    allimg = []
    for img in images:
        img = cv2.resize(img, (168, 48))
        img = np.reshape(img, (48, 168, 3))

        # normalize
        img = img.astype(np.float32)
        img = (img / 255. - mean_value) / std_value
        allimg.append(img)
    allimg = np.array(allimg)
    allimg = allimg.transpose([0, 3, 1, 2])
    allimg = torch.from_numpy(allimg)

    allimg = allimg.to(device)
    #allimg = allimg.view(1, *img.size())
    return allimg

def quchong(S):
  str1=[""]
  for i in S:
      if i == str1[-1]:
          str1.pop()
      else:
          str1.append(i)
  return ''.join(str1)
def get_plate_result(img, device, model):
    input = image_processing(img, device)
    preds = model(input)
    # print(preds)
    preds = preds.detach().cpu().numpy()
    newPreds=plate_Name[preds]
    allpreds=[]
    for every in newPreds:
        everys = np.append(every[1:], '0')
        c = every[every != everys]
        c=''.join(list(np.delete(c,np.where(c=='#'))))
        if c!='苏C8':
            allpreds.append(c)
    # if not (plate[0] in plateName[1:44] ):
    #     return ""
    return  allpreds


def init_model(device, model_path):
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    model_path = os.sep.join([sys.path[0], model_path])
    model = myNet_ocr(num_classes=78, export=True, cfg=cfg)

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


# model = init_model(device)
if __name__ == '__main__':

    image_path = "images/tmp2424.png"
    testPath = r"double_plate"
    fileList = []
    allFilePath(testPath, fileList)
    #    result = get_plate_result(image_path,device)
    #    print(result)
    model = init_model(device)
    right = 0
    begin = time.time()
    for imge_path in fileList:
        plate = get_plate_result(imge_path)
        plate_ori = imge_path.split('/')[-1].split('_')[0]
        # print(plate,"---",plate_ori)
        if (plate == plate_ori):

            right += 1
        else:
            print(plate_ori, "--->", plate, imge_path)
    end = time.time()
    print("sum:%d ,right:%d , accuracy: %f, time: %f" % (len(fileList), right, right / len(fileList), end - begin))

