from PIL import Image
import face_recognition
image = face_recognition.load_image_file('temp/1612410601.3614488.jpg')
face_locations = face_recognition.face_locations(image)

# face_locations =face_recognition.

# face_locations(image,number_of_times_to_upsample=0,model='cnn')

print('i found {} face(s) in this photograph.'.format(len(face_locations)))
for face_location in face_locations:
    top, right, bottom, left = face_location
    print('A face is located at pixel location Top:{},Left:{},Bottom:{},Right:{}'.format(top, right, bottom, left))
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()


