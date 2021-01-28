import cv2
import numpy as np

## 얼굴 100장을 찍어 그레이로 변환후 특정폴더에 저장하는 코드

#얼굴 인식용  xml 파일
face_classifier = cv2.CascadeClassifier('C:/Users/Lee/PycharmProjects/fr_ex4/haarcascade_frontalface_default.xml')

#전체 사진에서 얼굴 부위만 잘라 리턴
def face_extractor(img):
        #흑백처리
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #얼굴 찾기
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        #찾는 얼굴이 없으면 None으로 리턴
        if faces is():
            return None
        #있으면 해당 얼굴 크기만큼 croppped_face에 잘라 넣기
        # 얼굴이 2개 이상이면 가장 마지막의 얼굴만 남는다
        for(x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        #cropped_face 리턴
        return cropped_face

#노트북 내장카메라 실행
cap = cv2.VideoCapture(0)
#저장할  이미지 카운트 변수
count = 0

while True:
    #Capture frame-by-frame
    ret, frame = cap.read()
    #얼굴 감지하여 얼굴만 가져오기
    if face_extractor(frame) is not None:
        count+=1
        #얼굴 이미지 크기를 200x200으로 조정
        face = cv2.resize(face_extractor(frame),(200,200))
        #조정된 이미지를 흑백으로 변환
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #원하는 폴더에 jpg 파일로 저장하기 위해 경로설정
        file_name_path ='faces/user'+str(count)+'.jpg'
        #이미지 저장하기
        cv2.imwrite(file_name_path,face)

        #화면에 얼굴과 현재 저장 개수 표시
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not Found")
        pass

    #ENTER키는 13 esc는 27
    if cv2.waitKey(1) == 13 or count == 500:
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting Samples Complete!')