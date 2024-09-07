import cv2
import easyocr

# EasyOCR 리더 생성 (영어만 인식하도록 설정)
reader = easyocr.Reader(['en'])

# 웹캠으로부터 비디오 캡처 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        print("카메라에서 영상을 가져올 수 없습니다.")
        break

    # EasyOCR로 텍스트 인식
    result = reader.readtext(frame)

    # 결과 출력 및 이미지에 텍스트 표시
    for (bbox, text, prob) in result:
        # bbox는 인식된 텍스트의 좌표, text는 인식된 텍스트, prob는 확률
        top_left = tuple([int(val) for val in bbox[0]])
        bottom_right = tuple([int(val) for val in bbox[2]])
        frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        frame = cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 프레임을 보여줌
    cv2.imshow('Webcam - EasyOCR', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()