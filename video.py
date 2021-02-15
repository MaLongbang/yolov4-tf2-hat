# -------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
# -------------------------------------#
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from yolo import YOLO
import face_recognition
import os
import time
import shutil
import threading


def face_recognition_hat():
    temps = os.listdir('temp')
    face_images = os.listdir('face_database')
    result_faces_path = 'result_face/' + str(time.strftime('%Y%m%d', time.localtime(time.time())))
    for temp in temps:
        temp_time = temp.split('.')[0]

        # 读出缓存区的人脸照片转换成encoded格式
        image_to_be_matched = face_recognition.load_image_file('temp/' + temp)
        image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)
        result = {}

        # 如果存在继续执行，不存在就跳出本次循环
        if image_to_be_matched_encoded:
            image_to_be_matched_encoded = image_to_be_matched_encoded[0]
        else:
            shutil.copyfile('temp/' + temp, result_faces_path + '/' + 'unknow' + '_' + temp_time + '.jpg')
            os.remove('temp/' + temp)
            continue
        # 导入本地的人脸库进行对比 返回欧式距离最小的人脸 id
        for face_image in face_images:

            face_image_to_matched = face_recognition.load_image_file('face_database/' + face_image)
            face_image_to_matched_encode = face_recognition.face_encodings(face_image_to_matched)

            if face_image_to_matched_encode:
                face_image_to_matched_encode = face_image_to_matched_encode[0]
                compare_distance = face_recognition.face_distance([image_to_be_matched_encoded],
                                                                  face_image_to_matched_encode)
                result[face_image.split('.')[0]] = compare_distance[0]

            else:

                continue

        face_result = min(result, key=result.get)

        # 对结果库的数据进行对比 如果五分钟之内存在相同就跳过不保存
        if os.path.exists(result_faces_path):
            result_faces = os.listdir(result_faces_path)
            flag = 1
            for result_face in result_faces:
                user_id = result_face.split('.')[0].split('_')[0]
                user_time = result_face.split('.')[0].split('_')[1]
                if user_id == face_result and int(temp_time) - int(user_time) < 300000:
                    flag = 0
                    break
            if flag == 1:
                shutil.copyfile('temp/' + temp, result_faces_path + '/' + face_result + '_' + temp_time + '.jpg')

        else:
            os.makedirs(result_faces_path)
            shutil.copyfile('temp/' + temp, result_faces_path + '/' + face_result + '_' + temp_time + '.jpg')
        os.remove('temp/' + temp)


def video_com():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YOLO()
    # -------------------------------------#
    #   调用摄像头
    capture = cv2.VideoCapture("ttt.mp4")
    # -------------------------------------#
    # capture=cv2.VideoCapture(0)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 0.0
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output.mp4', fourcc, 24, (width, height))
    a = 0
    while True:
        flag = os.listdir('temp')
        if flag and a % 20 == 0:
            threading.Thread(target=face_recognition_hat).start()
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if ref:
            # 格式转变，BGRtoRGB
            a = a + 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            capture.release()
            out.release()
            break
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        if a % 20 == 0 or a == 1:
            frame = np.array(yolo.detect_image(frame))
        else:
            frame = np.array(frame)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        cv2.imshow("video", frame)
        c = cv2.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break
    print('视频总帧数', a)


if __name__ == '__main__':
    video_com()
