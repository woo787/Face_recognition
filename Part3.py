import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
##Part1과 part2를 합친내용이지만
##Part1을 통해 사진을갖고있지 않으면 이 파일은 실행되지 않습니다.
##Part1의 단점 : 사람이 옷을 입고 있다는 가정 하에 얼굴의 색을통해 구분하는 필터를 사용하다 보니 노출이 많은 옷을 입거나
## 피부색과 비슷한 물체가 있다면 얼굴을 찍어서 저장하지 않고 그 물체를 저장하는 문제가 발생
##단점 2 : 여러명은 아직까지 불가
##단점 3 : 정면이 아니라면 인식이 잘 되지 않음

##### 여기서부터는 Part2.py와 동일

data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is None:
        continue
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

#훈련할 데이터가 없다면 종료
if len(Labels) == 0:
    print("There is no data to train.")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!!!")

#### 여기까지 Part2.py와 동일

#### 여긴 Part1.py와 거의 동일
face_classifier = cv2.CascadeClassifier('C:/Users/Lee/PycharmProjects/fr_ex4/haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

#### 여기까지 Part1.py와 거의 동일
#카메라 열기
cap = cv2.VideoCapture(0)
while True:
    #카메라로 부터 사진 한장 읽기
    ret, frame = cap.read()
    # 얼굴 검출 시도
    image, face = face_detector(frame)
    try:
        #검출된 사진을 흑백으로 변환
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #위에서 학습한 모델로 예측시도
        result = model.predict(face)
        #result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.
        if result[1] < 500:
            #화면에 정확도를 0~100로 나타내기 위해 변수 confidence 지정
            confidence = int(100*(1-(result[1])/300))
            # 유사도 화면에 표시
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        #75 보다 크면 동일 인물로 간주해 UnLocked!
        if confidence > 75:
            cv2.putText(image, "Welcome, my master!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Face Recognizer', image)
        else:
           #75 이하면 타인.. Locked!!!
            cv2.putText(image, "Unauthorized personnel!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognizer', image)
    except:
        #얼굴 검출 안됨
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Recognizer', image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()