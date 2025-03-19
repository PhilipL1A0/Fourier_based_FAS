import cv2
import dlib
import numpy as np

class FaceDection(object):
    '''
    提供多种人脸检测方法的实现
    '''

    def __init__(self, model_name, face_max=True):
        '''
        :param model_name: 选择人脸检测的模型
        :param face_max: 返回的人脸数目,是最大的人脸还是所有人脸.目前该功能还未实现.默认只返回最大的人脸
        '''
        self.model_name = model_name
        self.face_max = face_max

        if model_name == "CAFFE":
            modelFile = "../models/FD/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "../models/FD/deploy.prototxt"
            net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
            self.model = net
        elif model_name == "TF":
            modelFile = "../models/FD/opencv_face_detector_uint8.pb"
            configFile = "../models/FD/opencv_face_detector.pbtxt"
            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
            self.model = net
        else:
            self.face_cascade = cv2.CascadeClassifier('../models/FD/haarcascade_frontalface_default.xml')

        self.conf_threshold = 0.7

    def face_detect(self, img, display=False):
        '''
        输入人脸,返回人脸照片
        :param img:
        :param display:
        :return:
        '''

        if self.model_name == "cv2":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))

            # 分析
            if len(faces) == 0:
                return None
            else:

                # 求最大人脸
                face_max = [0, 0, 0, 0]
                for i in range(len(faces)):
                    face = faces[i]
                    if face[2] > face_max[2]:
                        face_max = face

                # 人脸截取
                left = face_max[0]
                top = face_max[1]
                right = left + face_max[2]
                bottom = top + face_max[2]
                face_img = img[top:bottom, left:right]
        else:

            frameOpencvDnn = img.copy()
            frameHeight = frameOpencvDnn.shape[0]
            frameWidth = frameOpencvDnn.shape[1]
            img_mean = np.mean(frameOpencvDnn, (0, 1))

            # 数据预处理
            blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), img_mean, False, False)

            self.model.setInput(blob)
            detections = self.model.forward()
            bboxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            if display:
                cv2.imshow("frame", frameOpencvDnn)
                cv2.waitKey(0)

            # 如果检测到多个人脸,取最大的人脸
            if len(bboxes) > 1:
                bbox_max = [0, 0, 0, 0]
                bbox_max_len = bbox_max[2] - bbox_max[0]
                for bbox in bboxes:
                    if (bbox[2] - bbox[0]) > bbox_max_len:
                        bbox_max = bbox
                        bbox_max_len = bbox_max[2] - bbox_max[0]

                face_img = img[bbox_max[1]:bbox_max[3], bbox_max[0]:bbox_max[2]]
            elif len(bboxes) == 0:
                '''检测不到人脸'''
                return None
            else:
                bbox_max = bboxes[0]
                face_img = img[bbox_max[1]:bbox_max[3], bbox_max[0]:bbox_max[2]]

        return face_img


class LandmarksDetection(object):
    '''
    提供了人脸关键点检测算法
    '''

    def __init__(self):
        PREDICTOR_PATH = '../models/FD/shape_predictor_68_face_landmarks.dat'
        self.a_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    def landmarks_detect(self, img_rgb, display):
        '''
        输入人脸,检测关键点,最好是人脸检测之后的人脸图像.
        :param img:
        :return:
        '''
        img_shape = img_rgb.shape
        # 类型转变，opencv_to_dlib
        x1 = 0
        y1 = 0
        x2 = x1 + img_shape[0]
        y2 = y1 + img_shape[1]
        rect = dlib.rectangle(x1, y1, x2, y2)

        img_key = img_rgb.copy()
        predictor = self.a_predictor
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        points_keys = []

        # 特征点检测,只取第一个,也就是最大的一个
        landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])

        # 特征点提取,标注
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            points_keys.append([point[0, 0], point[0, 1]])
            cv2.circle(img_key, pos, 2, (255, 0, 0), -1)

        if display:
            cv2.imshow("landmark", img_key)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("landmark display")
        self.a_landmark = points_keys
        return points_keys