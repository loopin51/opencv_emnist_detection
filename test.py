import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
# 1. CIFAR-100 데이터셋 로드 (스트리밍 방식)
(ds_train, ds_test), ds_info = tfds.load(
    'cifar100',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,  # (image, label) 튜플로 반환
    with_info=True,
)

# CIFAR-100의 클래스 이름 로드
cifar100_labels = ds_info.features['label'].names

# 데이터 전처리 함수 정의
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # 이미지 정규화
    label = tf.one_hot(label, ds_info.features['label'].num_classes)  # 원-핫 인코딩
    return image, label

# 데이터 파이프라인 구성
batch_size = 32

train_ds = ds_train.map(preprocess).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 2. CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(ds_info.features['label'].num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 요약
model.summary()

# 3. 모델 학습
model.fit(train_ds, epochs=20, validation_data=test_ds)

# 모델 저장
model.save('cifar100_cnn_model_streaming.h5')
print("Model saved as cifar100_cnn_model_streaming.h5")

# 4. 실시간 웹캠을 이용한 물체 인식

# 저장된 모델 로드
model = load_model('cifar100_cnn_model_streaming.h5')

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    """실시간으로 입력된 프레임을 모델에 입력할 수 있도록 전처리"""
    img = cv2.resize(frame, (32, 32))  # CIFAR-100 이미지 크기에 맞게 조정
    img = img.astype('float32') / 255.0  # 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 전처리
    input_img = preprocess_frame(frame)

    # 모델 예측
    predictions = model.predict(input_img)
    predicted_class = np.argmax(predictions, axis=1)
    class_name = cifar100_labels[predicted_class[0]]
    confidence = np.max(predictions) * 100

    # 결과를 프레임 위에 표시
    text = f'{class_name}: {confidence:.2f}%'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 프레임을 표시
    cv2.imshow('Real-time Object Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()