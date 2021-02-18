import cv2
from face_net import *
from inception_resnet_v1 import *
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
import pickle

def extract_face(filename, required_size=(160, 160)):
    img = filename
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(filename)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = filename[y1:y2, x1:x2]
    scale_y = img.shape[0] / 640
    scale_x = img.shape[1] / 640
    color = (255,0,0)
    thickness = 2
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    img = cv2.resize(img,(640,640))
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array, img ,scale_y, scale_x, x1, y1

img = cv2.imread('ben.jpg')
img, img_1, scale_y, scale_x, x, y = extract_face(img)
x = int(x / scale_x)
y = int(y / scale_y)
model_1 = InceptionResNetV1(weights_path='facenet_keras_weights.h5')
img = get_embedding(model_1, img)
img = asarray(img)
img = expand_dims(img, axis=0)


data = load('data-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
model = pickle.load(open('finalized_model.sav', 'rb'))
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
result = model.predict(img)
probability = model.predict_proba(img)
# get name
class_index = result[0]
class_probability = probability[0,class_index] * 100
predict_names = out_encoder.inverse_transform(result)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

cv2.putText(img_1, predict_names[0], (x, y-10), fontFace= cv2.FONT_HERSHEY_SIMPLEX , fontScale= 0.5, color=(255,0,0), thickness= 1)
cv2.imshow('image', img_1)
cv2.waitKey(0)

