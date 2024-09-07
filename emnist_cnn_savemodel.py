import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import MobileNetV2
import tensorflow_datasets as tfds

# EMNIST 데이터셋 로드
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# 데이터 전처리 함수 정의
def preprocess(image, label):
    image = tf.image.resize(image, (96, 96))  # MobileNetV2는 96x96 이상 크기 필요
    image = tf.cast(image, tf.float32) / 255.0  # 0-1로 정규화
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)  # 1채널을 3채널로 변환
    label = label - 1  # 레이블을 0-25 범위로 맞춤 (A-Z)
    return image, label

# 데이터 전처리 적용
ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# 배치 크기 설정
BATCH_SIZE = 32

# 캐싱을 제거하고, 프리페칭(prefetch)만 사용하여 메모리 최적화
ds_train = ds_train.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# 데이터 증강 레이어 추가 (텐서플로우 내장)
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

# MobileNetV2 기반 모델 로드
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

# 모델 구조 설정
model = Sequential([
    data_augmentation,  # 데이터 증강 레이어 추가
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')  # 알파벳 26개 분류
])

# 기존 레이어는 고정 (첫 번째 학습 단계에서)
base_model.trainable = False

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 (초기 단계: 상위 레이어만 학습)
model.fit(ds_train, epochs=10, validation_data=ds_test)

# 미세 조정 단계: 모델의 상위 레이어를 일부 학습 가능하도록 설정
base_model.trainable = True

# 학습률을 낮게 설정하여 미세 조정
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 (미세 조정 단계)
model.fit(ds_train, epochs=10, validation_data=ds_test)

# 모델 저장 (SavedModel 포맷으로 저장)
model.save('saved_model_format', save_format='tf')