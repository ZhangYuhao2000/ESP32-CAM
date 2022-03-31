import cv2
import urllib.request
import numpy as np
import time

kernel1 = np.ones((2, 2), np.uint8)
kernel2 = np.ones((3, 3), np.uint8)

col_list = {'red': {'Lower': np.array([156, 43, 46]), 'Upper': np.array([180, 255, 255])},  # 深红
            'pink': {'Lower': np.array([0, 43, 46]), 'Upper': np.array([10, 255, 255])},  # 桃红
            'blue': {'Lower': np.array([100, 43, 46]), 'Upper': np.array([124, 255, 255])},  # 蓝
            'green': {'Lower': np.array([35, 43, 46]), 'Upper': np.array([90, 255, 255])},  # 绿
            'purple': {'Lower': np.array([125, 43, 46]), 'Upper': np.array([155, 255, 255])},  # 紫
            }

choose_col = 'green'
url = 'http://192.168.43.52/cam-lo.jpg'


def gamma_trans(gamimg, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(gamimg, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def img_processing(input_img, measure, output_res, out1, out2, ordy):
    newimg = cv2.GaussianBlur(input_img, (3, 3), 1)
    newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
    ofnewing = newimg.copy()

    ret, thresh = cv2.threshold(newimg, measure, 255, cv2.THRESH_BINARY)  # 转换为二值图
    thresh = cv2.dilate(thresh, kernel2, iterations=4)
    thresh = cv2.erode(thresh, kernel2, iterations=3)  # iterations为迭代次数
    imgup = thresh[0:60, 0:320]  # 截取部分图像（ROI方法）
    mask = np.zeros(thresh.shape[:2], np.uint8)
    mask[60:240, 0:320] = 255
    imgdown = cv2.bitwise_and(thresh, thresh, mask=mask)  # 与操作
    contours, hierarchy = cv2.findContours(imgdown, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # 注意版本，有点版本返回值为3个参数（有原图）
    draw_img = input_img.copy()  # 需要复制，绘制会改变原图像
    draw_img1 = input_img.copy()
    res1 = cv2.drawContours(draw_img1, contours, -1, (0, 0, 255), 1)  # 绘制边界

    for (i, c) in enumerate(contours):
        if cv2.arcLength(c, True) > 600:
            (x, y, w, h) = cv2.boundingRect(c)
            output_res = cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 绘制矩形判定框
            cv2.line(output_res, (60, 240), (x, y), (255, 255, 0), 3)
            cv2.line(output_res, (260, 240), (x + w, y), (255, 255, 0), 3)
            cv2.line(output_res, (160, 240), (int(x + w / 2), 200), (255, 100, 100), 3)
            if 140 < x + w / 2 < 180:
                ordy = 1
            elif 140 > x + w / 2:
                ordy = 3
            elif 180 < x + w / 2:
                ordy = 4
        else:
            ordy = 0
            output_res = draw_img1

    contours2, hierarchy2 = cv2.findContours(imgup, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)  # 注意版本，有点版本返回值为3个参数（有原图）
    res1 = cv2.drawContours(draw_img1, contours2, -1, (0, 255, 255), 1)  # 绘制边界
    for (i, c) in enumerate(contours2):
        (x, y, w, h) = cv2.boundingRect(c)
        output_res = cv2.rectangle(draw_img, (x, y), (x + w, y + h), (255, 255, 0), 1)  # 绘制矩形判定框

    out1 = imgdown
    out2 = res1

    return output_res, out1, out2, ordy


def automatic_drive(input_img, times, output_res, out1, out2, ordy, ):
    ordy = 0

    v1 = cv2.Canny(input_img, 100, 150)
    har_img = input_img.copy()

    # 低通滤波部分
    # B = input_img[:, :, 0]  # 采集通道信息
    # G = input_img[:, :, 1]
    # R = input_img[:, :, 2]
    #
    # grayimg = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)  # 取灰度图
    #
    # img_float32 = np.float32(grayimg)
    # dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    # dft_shift = np.fft.fftshift(dft)
    # rows, cols = grayimg.shape
    # crow, ccol = int(rows / 2), int(cols / 2)
    #
    # mask = np.ones((rows, cols, 2), np.uint8)
    # mask[crow - 100:crow + 100, ccol - 100:ccol + 100] = 0
    #
    # fshift = dft_shift * mask
    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = cv2.idft(f_ishift)
    # img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    #
    # g = img_back[:]
    # p = 0.2989
    # q = 0.5870
    # t = 0.1140
    # B_new = (g - p * R - q * G) / t
    # B_new = np.uint8(B_new)
    # newimg = np.zeros((input_img.shape)).astype("uint8")
    # newimg[:, :, 0] = B_new
    # newimg[:, :, 1] = G
    # newimg[:, :, 2] = R

    # hsv变换部分
    img_hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(har_img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    input_img[dst > 0.01 * dst.max()] = [0, 0, 255]
    img_hsv[dst > 0.01 * dst.max()] = [0, 0, 255]

    output_res = input_img
    out1 = img_hsv
    out2 = v1

    return output_res, out1, out2, ordy


def target_follow(input_img, times, output_res, out1, out2, ordy, tracker, centerx, centery):
    ori_img = cv2.bilateralFilter(input_img, 13, 30, 11)
    out1 = input_img.copy()
    ordy = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        bbox = (287, 23, 86, 320)
        bbox = cv2.selectROI(ori_img, False)
        ok = tracker.init(ori_img, bbox)
    else:
        ok, bbox = tracker.update(ori_img)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        centerx, centery = (int(bbox[0]) + int(bbox[0] + bbox[2])) / 2, (int(bbox[1]) + int(bbox[1] + bbox[3])) / 2
        cv2.rectangle(ori_img, p1, p2, (255, 0, 0), 2, 1)

        # print('centerx=', centerx, 'centery=', centery)

        cv2.line(out1, (30, 240), (100, 120), (0, 200, 250), 3)
        cv2.line(out1, (100, 120), (220, 120), (0, 200, 250), 3)
        cv2.line(out1, (220, 120), (290, 240), (0, 200, 250), 3)
        cv2.line(out1, (160, 240), (160, 140), (0, 200, 250), 3)
        cv2.line(out1, (120, 220), (200, 220), (0, 200, 250), 3)
        cv2.line(out1, (125, 190), (195, 190), (0, 200, 250), 3)
        cv2.line(out1, (120, 160), (200, 160), (0, 200, 250), 3)
        cv2.putText(out1, "30", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 250), 2)
        cv2.putText(out1, "60", (200, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 250), 2)
        cv2.putText(out1, "90", (200, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 250), 2)

        output_res = ori_img
        out2 = ori_img

    return output_res, out1, out2, ordy, centerx, centery


def terrace_move(input_img, measure, x, y, xround, pwmval, outround, outpwmval):
    center_x = 80
    center_y = 60
    if x == 0 or y == 0:
        outround = xround
        outpwmval = pwmval
    else:
        dx = x - center_x
        dy = y - center_y
        xg = 0.23
        yg = 7.22

        outround = xround + int((x - center_x) / 2.1)
        if outround > 1560:
            outround = outround - 1560
        elif outround < 0:
            outround = outround + 1560

        outpwmval = pwmval - int((y - center_y) / 31)
        if outpwmval > 26:
            outpwmval = 26
        elif outpwmval < 0:
            outpwmval = 0
        print("ox: ", outround)  # 打印发送信息
        print("op: ", outpwmval)  # 打印发送信息

        outround = outround + 20000
        outpwmval = (outpwmval + 10) * 10 + 30000

    return outround, outpwmval


def picture_processing(img_input):
    res = img_input.copy()
    res = cv2.GaussianBlur(res, (5, 5), 2)
    res = cv2.bilateralFilter(res, 5, 27, 19)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = 255 * np.power(res / 255, 0.39)
    res = np.around(res)
    res[res > 255] = 255
    res = res.astype(np.uint8)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    # res = clahe.apply(res)

    return res


def catface_detect(img_org, img_input, classfier, center_x, center_y):
    img_change = img_org.copy()
    # h, w = img_input.shape
    # print("h: ", h, "   w: ", w)  # 打印发送信息
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res = clahe.apply(img_input)
    fiimg = cv2.copyMakeBorder(res, 0, 0, 0, 160, cv2.BORDER_REFLECT, 0)
    facerects = classfier.detectMultiScale(res, scaleFactor=1.1, minNeighbors=4)
    if len(facerects) > 0:  # 大于0则检测到人脸
        for faceRect in facerects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            cv2.rectangle(img_change, (x, y), (x + w, y + h), (20, 170, 20), 2)
            center_x = (x + w + x) / 2
            center_y = (y + h + y) / 2
            cv2.putText(img_change, "face_front", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 170, 20), 2)
            print("center_x: ", center_x, "   center_y: ", center_y)  # 打印发送信息

    v1 = cv2.Canny(img_input, 100, 150)
    # gray = np.float32(res)
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    # img_change[dst > 0.01 * dst.max()] = [0, 0, 255]

    out = res
    out1 = img_change
    out2 = v1
    return out, out1, out2, center_x, center_y


def face_detect(img_org, img_input, classfier, catclassfier, center_x, center_y):
    img_change = img_org.copy()
    # h, w = img_input.shape
    # print("h: ", h, "   w: ", w)  # 打印发送信息
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res = clahe.apply(img_input)
    fiimg = cv2.copyMakeBorder(res, 0, 0, 0, 160, cv2.BORDER_REFLECT, 0)
    facerects = classfier.detectMultiScale(res, scaleFactor=1.1, minNeighbors=4)
    profile_facerects = catclassfier.detectMultiScale(fiimg, scaleFactor=1.1, minNeighbors=4)
    if len(facerects) > 0:  # 大于0则检测到人脸
        for faceRect in facerects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            cv2.rectangle(img_change, (x, y), (x + w, y + h), (20, 170, 20), 2)
            center_x = (x + w + x) / 2
            center_y = (y + h + y) / 2
            cv2.putText(img_change, "face_front", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 170, 20), 2)
            print("center_x: ", center_x, "   center_y: ", center_y)  # 打印发送信息

    if len(profile_facerects) > 0:  # 大于0则检测到人脸
        for catfaceRect in profile_facerects:  # 单独框出每一张人脸
            x, y, w, h = catfaceRect
            if x > 160:
                x = 320 - (x + h)
            cv2.rectangle(img_change, (x, y), (x + w, y + h), (20, 20, 170), 2)
            center_x = (x + w + x) / 2
            center_y = (y + h + y) / 2
            cv2.putText(img_change, "face_profile", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 170), 2)
            print("center_x: ", center_x, "   center_y: ", center_y)  # 打印发送信息

    v1 = cv2.Canny(img_input, 100, 150)
    # gray = np.float32(res)
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    # img_change[dst > 0.01 * dst.max()] = [0, 0, 255]

    out = res
    out1 = img_change
    out2 = v1
    return out, out1, out2, center_x, center_y


def target_match(img_input, target):
    h, w = target.shape[:2]
    res = cv2.matchTemplate(img_input, target, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # 最小值， 最大值， 最小值位置， 最大值位置

    threshold = 0.5  # 指定阈值
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(img_input, pt, bottom_right, (0, 0, 255), 1)

    return img_input


def cube_detect(img_input, img_org, cx, cy):
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
    res = img_input

    return res, img_org


# 鱼眼有效区域截取
def cut(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(cnts)
    r = max(w / 2, h / 2)
    # 提取有效区域
    img_valid = img[y:y + h, x:x + w]
    return img_valid, int(r)


# 鱼眼矫正
def undistort(src, r):
    # r： 半径， R: 直径
    R = 2 * r
    # Pi: 圆周率
    Pi = np.pi
    # 存储映射结果
    dst = np.zeros((R, R, 3))
    src_h, src_w, _ = src.shape

    # 圆心
    x0, y0 = src_w // 2, src_h // 2

    # 数组， 循环每个点
    range_arr = np.array([range(R)])

    theta = Pi - (Pi / R) * range_arr.T
    temp_theta = np.tan(theta) ** 2

    phi = Pi - (Pi / R) * range_arr
    temp_phi = np.tan(phi) ** 2

    tempu = r / (temp_phi + 1 + temp_phi / temp_theta) ** 0.5
    tempv = r / (temp_theta + 1 + temp_theta / temp_phi) ** 0.5

    # 用于修正正负号
    flag = np.array([-1] * r + [1] * r)

    # 加0.5是为了四舍五入求最近点
    u = x0 + tempu * flag + 0.5
    v = y0 + tempv * np.array([flag]).T + 0.5

    # 防止数组溢出
    u[u < 0] = 0
    u[u > (src_w - 1)] = src_w - 1
    v[v < 0] = 0
    v[v > (src_h - 1)] = src_h - 1

    # 插值
    dst[:, :, :] = src[v.astype(int), u.astype(int)]
    return dst


if __name__ == "__main__":
    t = time.perf_counter()
    frame = cv2.imread('../imgs/pig.jpg')
    cut_img, R = cut(frame)
    t = time.perf_counter()
    result_img = undistort(cut_img, R)
    cv2.imwrite('../imgs/pig_vector_nearest.jpg', result_img)
    print(time.perf_counter() - t)
