import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 학습된 SavedModel 형식의 CNN 모델 로드 (디렉토리 경로를 입력)
model = load_model('saved_model_format')  # 모델이 저장된 디렉토리 경로를 지정하세요

# 알파벳 레이블
alphabet_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# 글자를 예측하는 함수
def predict_letter(img):
    # 이미지를 모델이 요구하는 (96, 96) 크기로 변환
    img = cv2.resize(img, (96, 96))  # 이미지 크기를 (96, 96)으로 조정
    print(f"Image shape after resizing: {img.shape}")  # 크기 확인

    # 흑백 이미지일 경우 3채널로 변환
    if len(img.shape) == 2:  # 흑백 이미지일 경우만 변환
        img = np.stack((img,)*3, axis=-1)
        print(f"Image shape after channel conversion (grayscale to RGB): {img.shape}")  # 변환 후 크기 확인

    # 이미지를 (1, 96, 96, 3) 크기로 변환하여 모델에 입력
    img = img.reshape(1, 96, 96, 3)
    print(f"Image shape after reshaping: {img.shape}")  # 최종 크기 확인

    img = img / 255.0  # 정규화
    result = model.predict([img])  # 예측 수행
    return alphabet_labels[np.argmax(result)]

# 글자를 추출하는 함수 (간단한 세그멘테이션)
def segment_letters(image):
    # 이미지를 이진화
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 컨투어 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_images = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # 너무 작은 노이즈 제거
            letter = image[y:y+h, x:x+w]
            letter_images.append((x, letter))  # x 위치와 함께 저장 (글자 순서대로 정렬을 위해)

    # x 위치를 기준으로 글자 정렬
    letter_images = sorted(letter_images, key=lambda item: item[0])

    return [img for (_, img) in letter_images]

# 실시간 웹캡 캡처 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임의 크기와 차원 출력
    print(f"Captured frame shape: {frame.shape}")  # 컬러 이미지라면 (height, width, 3)이어야 함

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 관심 영역(ROI) 설정 (캡처 영역, 여기서는 고정된 사각형을 사용)
    roi = gray[100:400, 100:600]  # 단어가 있을 것으로 예상되는 부분
    cv2.rectangle(frame, (100, 100), (600, 400), (255, 0, 0), 2)

    # 단어를 구성하는 개별 글자 분리
    letters = segment_letters(roi)

    # 분리된 글자들을 하나씩 인식
    recognized_word = ''
    for letter_img in letters:
        letter = predict_letter(letter_img)
        recognized_word += letter

    # 인식된 단어 화면에 표시
    cv2.putText(frame, f"Word: {recognized_word}", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow("Handwritten Word Recognition", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()