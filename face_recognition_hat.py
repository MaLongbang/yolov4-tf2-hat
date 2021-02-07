import face_recognition
import os
import time
import shutil

temps = os.listdir('temp')
face_images = os.listdir('face_database')

for temp in temps:
    temp_time = temp.split('.')[0]

    # 读出缓存区的人脸照片转换成encoded格式
    image_to_be_matched = face_recognition.load_image_file('temp/'+temp)
    image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)
    result = {}
    # 如果存在继续执行，不存在就跳出本次循环
    if image_to_be_matched_encoded:
        image_to_be_matched_encoded = image_to_be_matched_encoded[0]

    else:
        os.remove('temp/' + temp)
        continue
    # 导入本地的人脸库进行对比 返回欧式距离最小的人脸 id
    for face_image in face_images:

        face_image_to_matched = face_recognition.load_image_file('face_database/'+face_image)
        face_image_to_matched_encode = face_recognition.face_encodings(face_image_to_matched)

        if face_image_to_matched_encode:
            face_image_to_matched_encode = face_image_to_matched_encode[0]
            compare_distance = face_recognition.face_distance([image_to_be_matched_encoded],
                                                              face_image_to_matched_encode)
            result[face_image.split('.')[0]] = compare_distance[0]

        else:
            continue

    face_result = min(result, key=result.get)
    result_faces_path = 'result_face/'+str(time.strftime('%Y%m%d', time.localtime(time.time())))

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
        shutil.copyfile('temp/'+temp, result_faces_path+'/'+face_result+'_'+temp_time+'.jpg')
    os.remove('temp/'+temp)

