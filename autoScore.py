import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,precision_score
import imutils
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model
import json
import cv2
import os
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 檢測設備並設置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用設備: {device}")

def load_model_safely(model_path, model_type, device):
    """安全地加載模型，處理設備轉換"""
    try:
        if model_type == "resnet":
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
        elif model_type == "vit":
            model = models.vit_b_16(pretrained=True)
            num_ftrs = model.heads.head.in_features
            model.heads.head = nn.Linear(num_ftrs, 2)
        else:
            raise ValueError(f"不支持的模型類型: {model_type}")
        
        # 加載模型權重，確保正確的設備映射
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"加載模型 {model_path} 時發生錯誤: {str(e)}")
        raise

def resnet(imgpath, modelpath):
    try:
        model2 = load_model_safely(modelpath, "resnet", device)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((100, 100)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        img = imgpath
        data = transform(img)
        data = torch.unsqueeze(data, dim=0)
        data = data.to(device)
        
        with torch.no_grad():
            outputs = model2(data)
            _, predicted = torch.max(outputs, 1)
        return predicted
    except Exception as e:
        logger.error(f"resnet 處理時發生錯誤: {str(e)}")
        raise

def resnet5(imgpath, modelpath):
    try:
        model2 = load_model_safely(modelpath, "resnet", device)
        model2.fc = nn.Linear(model2.fc.in_features, 6)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((100, 100)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        img = imgpath
        data = transform(img)
        data = torch.unsqueeze(data, dim=0)
        data = data.to(device)
        
        with torch.no_grad():
            outputs = model2(data)
            _, predicted = torch.max(outputs, 1)
        return predicted
    except Exception as e:
        logger.error(f"resnet5 處理時發生錯誤: {str(e)}")
        raise

def CNN(imgpath, modelpath):
    try:
        model = load_model(modelpath, compile=False)
        img = imgpath
        img = img[:,:,0]
        img = cv2.resize(img, (56, 56), interpolation=cv2.INTER_CUBIC)
        testList = np.array([img])
        y_proba = model.predict(testList)
        y_pred = np.argmax(y_proba, axis=-1)
        return y_pred
    except Exception as e:
        logger.error(f"CNN 處理時發生錯誤: {str(e)}")
        raise

def Vit(imgpath, modelpath):
    try:
        model2 = load_model_safely(modelpath, "vit", device)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        img = imgpath
        data = transform(img)
        data = torch.unsqueeze(data, dim=0)
        data = data.to(device)
        
        with torch.no_grad():
            outputs = model2(data)
            _, predicted = torch.max(outputs, 1)
        return predicted
    except Exception as e:
        logger.error(f"Vit 處理時發生錯誤: {str(e)}")
        raise

modelList=[
    [
        {
            "path": "draw_trainingdata/O/圓/Oo_resnet50.pth",
            "modelType": resnet,
            "Reverse": True,
            "Score": 1
        },
        {
            "path": "draw_trainingdata/O/缺口/Oc_resnet50.pth",
            "modelType": resnet,
            "Reverse": True,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/O/突出/Ox_resnet50.pth",
            "modelType": resnet,
            "Reverse": False,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/O/有角/Od_resnet50.pth",
            "modelType": resnet,
            "Reverse": True,
            "Score": 0.2
        },
    ],
    [
        {
            "path": "draw_trainingdata/X/十字/Xx_resnet50.pth",
            "modelType": resnet,
            "Reverse": True,
            "Score": 2
        },
        {
            "path": "draw_trainingdata/X/對稱/Xt_resnet50.pth",
            "modelType": resnet,
            "Reverse": True,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/X/直線/Xz_resnet50.pth",
            "modelType": resnet,
            "Reverse": False,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/X/直角/Xv_vit_b_16.pth",
            "modelType": Vit,
            "Reverse": False,
            "Score": 0.2
        },
    ],
    [
        {
            "path": "draw_trainingdata/S/方形/Ss_resnet50.pth",
            "modelType": resnet,
            "Reverse": True,
            "Score": 4
        },
        {
            "path": "draw_trainingdata/S/缺口/Sc_CNN.h5",
            "modelType": CNN,
            "Reverse": False,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/S/突出/Sx_resnet50.pth",
            "modelType": resnet,
            "Reverse": False,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/S/直線/Sz_resnet50.pth",
            "modelType": resnet,
            "Reverse": False,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/S/直角/Sl_CNN.h5",
            "modelType": CNN,
            "Reverse": False,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/S/對邊相等/St_resnet50.pth",
            "modelType": resnet,
            "Reverse": False,
            "Score": 0.1
        },
    ],
    [
        {
            "path": "draw_trainingdata/T/三角/Tt_resnet50.pth",
            "modelType": resnet,
            "Reverse": True,
            "Score": 8
        },
        {
            "path": "draw_trainingdata/T/倒反/Tr_resnet50.pth",
            "modelType": resnet,
            "Reverse": True,
            "Score": 0
        },
        {
            "path": "draw_trainingdata/T/缺口/Tc_resnet50.pth",
            "modelType": resnet,
            "Reverse": True,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/T/突出/Tx_resnet50.pth",
            "modelType": resnet,
            "Reverse": False,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/T/直線/Tz_resnet50.pth",
            "modelType": resnet,
            "Reverse": False,
            "Score": 0.2
        },
    ],
    [
        {
            "path": "draw_trainingdata/D/菱形/Dd_CNN.h5",
            "modelType": CNN,
            "Reverse": False,
            "Score": 16
        },
        {
            "path": "draw_trainingdata/D/缺口/Dc_resnet50.pth",
            "modelType": resnet,
            "Reverse": False,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/D/突出/Dx_CNN.h5",
            "modelType": CNN,
            "Reverse": False,
            "Score": 0.2
        },
        {
            "path": "draw_trainingdata/D/直線/Dz_resnet50.pth",
            "modelType": resnet,
            "Reverse": False,
            "Score": 0.2
        },
    ],
    []
]

#特徵辨識
def ScoreA(imgpath,modellist):
    sc=0
    #shape
    print(modellist[0]["path"])
    res = modellist[0]["modelType"](imgpath, modellist[0]["path"]).item()
    if modellist[0]["Reverse"]:
        res = 1-res
    print(res)
    if res==1:
        return sc

    sc += modellist[0]["Score"]

    #others
    for i in range(1,len(modellist)):
        print(modellist[i]["path"])
        res = modellist[i]["modelType"](imgpath, modellist[i]["path"]).item()
        if modellist[i]["Reverse"]:
            res = 1-res
        print(res)
        if res==1:
            sc -= modellist[i]["Score"]
    return sc

def ScoreB(imgpath,modellist):
    sc=0
    #shape
    print(modellist[0]["path"])
    res = modellist[0]["modelType"](imgpath, modellist[0]["path"]).item()
    if modellist[0]["Reverse"]:
        res = 1-res
    print(res)
    if res==1:
        return sc
    #r
    print(modellist[1]["path"])
    res = modellist[1]["modelType"](imgpath, modellist[1]["path"]).item()
    if modellist[1]["Reverse"]:
        res = 1-res
    print(res)
    if res==1:
        return sc

    sc += modellist[0]["Score"]

    #others
    for i in range(2,len(modellist)):
        print(modellist[i]["path"])
        res = modellist[i]["modelType"](imgpath, modellist[i]["path"]).item()
        if modellist[i]["Reverse"]:
            res = 1-res
        print(res)
        if res==1:
            sc -= modellist[i]["Score"]
    return sc

def ScoreC(imgpath,modellist):
    print(0)
    return 0

ScoreList = [ScoreA, ScoreA, ScoreA, ScoreB, ScoreA, ScoreC]

#5p
def Score(data):
  res = CNN(data, "draw_trainingdata/five_pattern/five_p_CNN.h5")
  res = res[0]
  print("形狀: ", res)
  score = ScoreList[res](data, modelList[res])

  print("Score: ", round(score,4))
  return res, round(score,4)

#runall
import cv2
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import os

def Runall(data: np.ndarray):
    print("Now -->")
    imgList = []
    # img = io.imread("WETP000173.jpg")
    img = data
    mid = np.vstack(img)
    sort = np.sort(mid,axis=0)
    sort = sort[np.any(sort,axis=1)]
    midnum = sort[int(len(sort)/2)]

    blur = cv2.blur(img, (80, 80))
    lower = np.array([midnum[0]-40,midnum[1]-40,midnum[2]-40])
    upper = np.array([midnum[0]+40,midnum[1]+40,midnum[2]+40])
    mask = cv2.inRange(blur, lower, upper)
    paper,h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for i in range(len(paper)):
        areas.append(cv2.contourArea(paper[i]))
    max_id = areas.index(max(areas))

    i=10
    while True:
        paperapprox = cv2.approxPolyDP(paper[max_id],i,True)
        if len(paperapprox)>10:
            i+=10
        else:
            break
    pmask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    cv2.drawContours(pmask, [paperapprox], 0, (255,255,255), -1)
    kernel = np.ones(shape=[30,30],dtype=np.uint8)
    pmask = cv2.dilate(pmask, kernel, iterations=1)
    img = cv2.blur(img, (5, 5))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_and(img, img, mask = pmask )

    gray[gray==0]=0
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 腐蝕
    kernel = np.ones(shape=[3,3],dtype=np.uint8)
    erodeimg = cv2.erode(thresh,kernel=kernel)
    # 膨脹
    kernel = np.ones(shape=[3,3],dtype=np.uint8)
    dilateimg = cv2.dilate(erodeimg, kernel, iterations=1)

    # 膨脹
    kernel = np.ones(shape=[6,6],dtype=np.uint8)
    fc = cv2.dilate(erodeimg, kernel, iterations=1)

    fc = cv2.bitwise_and(fc, fc, mask = pmask )

    # 尋找輪廓
    cnt,h = cv2.findContours(fc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #去除面積過大的
    cnt = list(cnt)
    for i in range(len(cnt)-1,-1,-1):
        approx = cv2.approxPolyDP(cnt[i],3,True)
        rect = cv2.minAreaRect(approx)
        if rect[1][0]*rect[1][1] > img.shape[0]*img.shape[1]/7:
            cnt.pop(i)
            continue
        elif rect[1][0]*rect[1][1] < img.shape[0]*img.shape[1]/800:
            cnt.pop(i)
            continue

    #重疊合併
    cnt2 = []
    used = [False] * len(cnt)
    for i in range(len(cnt)):
        if used[i]:
            continue
        merged_elem = cnt[i]
        for j in range(i+1,len(cnt)):
            if used[j]:
                continue
            poly1 = Polygon(cnt[i].reshape((-1,2))).convex_hull
            poly2 = Polygon(cnt[j].reshape((-1,2))).convex_hull
            if poly1.intersects(poly2):
                merged_elem = np.append(merged_elem, cnt[j], axis=0)
                used[j] = True
        cnt2 += [merged_elem]

    cnt = tuple(cnt2)

    for i in range(len(cnt)):
        black = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
        approx = cv2.approxPolyDP(cnt[i],3,True)
        rect = cv2.minAreaRect(approx)

        line = rect[1][0]/rect[1][1]
        if line < 1:
            line = rect[1][1]/rect[1][0]
        if line > 6:
            continue

        maxbox = np.max(approx, axis=0)[0]
        minbox = np.min(approx, axis=0)[0]
        box = np.array([maxbox,[minbox[0],maxbox[1]],minbox,[maxbox[0],minbox[1]]])
        boxw = maxbox[0] - minbox[0]
        boxh = maxbox[1] - minbox[1]
        short_side = boxw
        if boxw>boxh:
            short_side = boxh

        cv2.drawContours(img, [approx], 0, (255,0,0), 10)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        #切圖
        target = dilateimg[minbox[1]:maxbox[1], minbox[0]:maxbox[0]]
        try:
            image = cv2.resize(target, (100, 100), interpolation=cv2.INTER_CUBIC)
            imgList += [image]
        except:
            print("error")
            continue
    O=0
    X=0
    S=0
    T=0
    D=0
    for img in imgList:
        img = np.stack((img,) * 3, axis=-1)
        p, s = Score(img)
        if p == 0 and s > O:
            O = s
        elif p == 1 and s > X:
            X = s
        elif p == 2 and s > S:
            S = s
        elif p == 3 and s > T:
            T = s
        elif p == 4 and s > D:
            D = s
    print("total: ", O+X+S+T+D)
    print(O, X, S, T, D)
    tmp = {"Score": O+X+S+T+D, "O": O, "X": X, "S": S, "T": T, "D": D}
    return tmp