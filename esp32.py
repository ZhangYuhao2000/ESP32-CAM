import cv2
import mode
import sys
import serial
import time
import numpy as np
import math
import urllib.request
from apscheduler.schedulers.background import BackgroundScheduler
import os
from time import sleep
import argparse

url = 'http://192.168.43.52/cam-lo.jpg'


class yolo_fast_v2():
    def __init__(self, objThreshold=0.3, confThreshold=0.3, nmsThreshold=0.4):
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split(
                '\n')  ###这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.stride = [16, 32]
        self.anchor_num = 3
        self.anchors = np.array(
            [12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87],
            dtype=np.float32).reshape(len(self.stride), self.anchor_num, 2)
        self.inpWidth = 352
        self.inpHeight = 352
        self.net = cv2.dnn.readNet('model.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                # confidences.append(float(confidence))
                confidences.append(float(confidence * detection[4]))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return frame

    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight))
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        outputs = np.zeros((outs.shape[0] * self.anchor_num, 5 + len(self.classes)))
        row_ind = 0
        for i in range(len(self.stride)):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(h * w)
            grid = self._make_grid(w, h)
            for j in range(self.anchor_num):
                top = row_ind + j * length
                left = 4 * j
                outputs[top:top + length, 0:2] = (outs[row_ind:row_ind + length,
                                                  left:left + 2] * 2. - 0.5 + grid) * int(self.stride[i])
                outputs[top:top + length, 2:4] = (outs[row_ind:row_ind + length,
                                                  left + 2:left + 4] * 2) ** 2 * np.repeat(
                    self.anchors[i, j, :].reshape(1, -1), h * w, axis=0)
                outputs[top:top + length, 4] = outs[row_ind:row_ind + length, 4 * self.anchor_num + j]
                outputs[top:top + length, 5:] = outs[row_ind:row_ind + length, 5 * self.anchor_num:]
            row_ind += length
        return outputs


# yolo_fast部分代码
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='img/000139.jpg', help="image path")
    parser.add_argument('--objThreshold', default=0.3, type=float, help='object confidence')
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.4, type=float, help='nms iou thresh')
    args = parser.parse_args()

model = yolo_fast_v2(objThreshold=args.objThreshold, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)

# 目标跟踪代码部分
(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

if __name__ == '__main__':

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = 'KCF'

    if int(minor_ver) < 3:
        tracker = cv2.legacy.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.legacy.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.legacy.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.legacy.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.legacy.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.legacy.TrackerGOTURN_create()

# 正常显示代码部分
m = 50
order = 0
cx = 160
cy = 120
xround = 0
pwmval = 174
com = 0
inuto = 2
xr = xround
pm = pwmval
ocr = 1
owm_xround = 0
owm_pwmval = 174
name = 0

# # 串口通信部分代码
# scale_percent = 80
# if __name__ == '__main__':
#     serial = serial.Serial('COM4', 115200, timeout=1)  # 设定串口4 波特率115200 超时计时1S
#     if serial.isOpen():
#         print("open success")
#     else:
#         print("open failed")
#
#
# # 串口数据发送任务
# def job():
#     if ocr == 2:
#         s2 = '%d' % xr
#         s3 = '%d' % pm
#     elif ocr == 1:
#         s2 = '%d' % owm_xround
#         s3 = '%d' % owm_pwmval
#
#     if com == 1:
#         data1 = ('#' + s2 + '!').encode()  # 待发送指令
#         serial.write(data1)  # 向串口发送
#         # print("send : ", data1)  # 打印发送信息
#     elif com == 2:
#         data2 = ('#' + s3 + '!').encode()  # 待发送指令
#         serial.write(data2)  # 向串口发送
#         # print("send : ", data2)  # 打印发送信息


def nothing(x):
    pass


classfier = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")  # 人脸识别调用文件
# catclassfier = cv2.CascadeClassifier("haarcascade_profileface.xml")  # 人脸识别调用文件

# scheduler = BackgroundScheduler()  # 调度器任务
# scheduler.add_job(job, 'interval', seconds=0.02)  # 调度器任务
# scheduler.start()  # 调度器任务

cv2.namedWindow('regulate')
cv2.namedWindow('cam')
cv2.createTrackbar('m', 'regulate', 10, 245, nothing)
cv2.createTrackbar('xround', 'regulate', 0, 1560, nothing)
cv2.createTrackbar('pwmval', 'regulate', 20, 26, nothing)

target = cv2.imread('target.jpg')

while 1:
    com = com + 1
    if com == 3:
        com = 1
    begin = time.time()
    imgResponse = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)  # 完成图片转换

    m = cv2.getTrackbarPos('m', 'regulate')
    owm_xround = cv2.getTrackbarPos('xround', 'regulate') + 20000
    owm_pwmval = (cv2.getTrackbarPos('pwmval', 'regulate') + 10) * 10 + 30000

    imgoff = mode.picture_processing(img)

    # res, new_out = mode.cube_detect(imgoff, img, cx, cy)
    # res = mode.target_match(img, target)
    # res, new_out, hes_out, cx, cy = mode.catface_detect(img, imgoff, classfier, cx, cy)
    # res, new_out, hes_out, order = mode.automatic_drive(out, inuto, out, out, out, order)
    # res, new_out, hes_out, order, cx, cy = mode.target_follow(img, inuto, img, img, img, order, tracker, cx, cy)
    # xr, pm = mode.terrace_move(img, m, cx, cy, owm_xround - 20000, (owm_pwmval - 30000) / 10 - 10, xround, pwmval)

    # yolo_fast目标检测
    img = cv2.bilateralFilter(img, 13, 30, 11)
    outputs = model.detect(img)
    frame = model.postprocess(img, outputs)

    end = time.time()
    timecos = end - begin
    cv2.putText(img, "FPS : " + str(int(1 / timecos)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 170, 20), 2)
    # cv2.putText(img, "ORD : " + str(int(order)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 170, 20), 2)

    cv2.imshow('cam', img)
    # cv2.imshow('0win1', res)
    # cv2.imshow('0win2', new_out)
    # cv2.imshow('0win3', hes_out)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('w'):
        ocr = 1
    elif tecla == ord('e'):
        ocr = 2
    elif tecla == ord('r'):
        name = name + 1
        cv2.imwrite('img\\' + str(name) + '.jpg', img)  # 保存图像
    elif tecla == 27:
        break

cv2.destroyAllWindows()
