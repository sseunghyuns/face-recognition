import dlib, cv2, sys
import numpy as np

# load video
cap = cv2.VideoCapture('boy.mp4') # 비디오 로드를 위한 VideoCapture. 파일 이름 대신 0을 넣으면 웹캠이 켜지고 내 얼굴로 테스트 가능
# load overlay image
overlay = cv2.imread('firefox.png', cv2.IMREAD_UNCHANGED) # cv2.IMREAD_UNCHANGED : file 이미지를 BGRA 타입으로 읽기. 알파 타입까지 읽을 수 있음.

scaler = 0.3 # 윈도우의 크기를 0.3으로 줄이자.
detector = dlib.get_frontal_face_detector() # 얼굴 디텍터 모듈 초기화
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 얼굴 특징점 모듈 초기화. 머신러닝으로 학습된 모델 파일 필요.

while True:
    ret, img = cap.read() # cap.read() : 동영상 파일에서 frame 단위로 읽기
    if not ret:
        break # frame이 없으면 프로그램 종료


    # img.shape[0] : 높이, img.shape[1] : 너비
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler))) # imshow 하기 전에 이미지를 줄여준다. 숫자형만 받으므로 int 적용.
    original  = img.copy() # 원본 이미지 저장

    # detect faces
    faces = detector(img)  #img에서 모든 얼굴 찾기
    face = faces[0] # 찾은 모든 얼굴에서 첫 번째 얼굴만 가져오기
    # face : 첫 번째 얼굴의 좌표. ex) [(234, 214) (413, 393)] 좌상, 우하 좌표 반환

    dlib_shape = predictor(img, face) # img의 face 영역안의 얼굴 특징점 찾기
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()]) # 계산하기 편하게 dlib 객체를 numpy 객체로 변환하여 shape_2d에 저장

    # compute boundaries and center of face
    top_left = np.min(shape_2d, axis = 0) # 좌상단 좌표. x는 왼쪽일수록, y 위로 갈수록 감소함. 각 행을 비교하여 가장 작은 값들을 반환.
    bottom_right = np.max(shape_2d, axis = 0) # 우하단 좌표

    face_size = int(max(bottom_right - top_left) * 1.8)  # 우하단에서 좌상단 좌표를 뺀 (x,y) 길이의 가장 긴 값

    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int) # 모든 특징점의 평균을 구해 얼굴의 중심 구하기. 소수점일수도 있으니 정수형으로 변환

    result = overlay_transparent(original, overlay, center_x, center_y, overlay_size=(face_size, face_size))

    # visualize
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color = (255,255,255), # 직사각형 그리기. 좌상단 : face.left(), face.top(), 우하단 : face.right(), face.bottom()
                        thickness=2, lineType=cv2.LINE_AA) # 색, 두깨, 선모양 지정.


    for s in shape_2d: # for loop 돌며 68개의 얼굴 특징점을 그려준다.
        cv2.circle(img, center = tuple(s), radius=1, color = (255, 255, 255), thickness=2, lineType= cv2.LINE_AA)

    # 좌상단, 우하단을 그려보자.
    cv2.circle(img, center=tuple(top_left), radius = 1 ,color = (255, 0, 0), thickness=2, lineType= cv2.LINE_AA) # 좌상단
    cv2.circle(img, center=tuple(bottom_right), radius = 1 ,color = (255, 0, 0), thickness=2, lineType= cv2.LINE_AA) # 좌상단

    # 얼굴의 중심 그리기
    cv2.circle(img, center=tuple((center_x, center_y)), radius = 1 ,color = (0, 0, 255), thickness=2, lineType= cv2.LINE_AA) # 좌상단


    cv2.imshow('img', img) # img라는 윈도우에 img 띄우기
    cv2.imshow('result',result)
    k = cv2.waitKey(1) & 0xFF # 1밀리 세컨드만큼 대기.
    if k == 27:
        break
