import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
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
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=-1)
    image = tf.tile(image, [1, 1, 3])  # 흑백 이미지를 3채널로 변환
    label = label - 1
    return image, label

# 데이터 전처리 적용
ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# 데이터 증강(Data Augmentation)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,   # 이미지 회전
    width_shift_range=0.1,  # 가로 이동
    height_shift_range=0.1,  # 세로 이동
    shear_range=0.1,     # 기울임
    zoom_range=0.1       # 확대/축소
)

# 배치 처리
BATCH_SIZE = 64
ds_train = ds_train.batch(BATCH_SIZE).map(lambda x, y: (datagen.flow(x, y, batch_size=BATCH_SIZE)))
ds_test = ds_test.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# MobileNetV2 기반 모델 로드
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

# 모델 구조 설정
model = Sequential([
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

# 학습률 스케줄러 설정: 성능이 향상되지 않으면 학습률을 감소시킴
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# 모델 학습 (초기 단계: 상위 레이어만 학습)
model.fit(ds_train, epochs=10, validation_data=ds_test, callbacks=[lr_scheduler])

# 미세 조정 단계: 모델의 상위 레이어를 일부 학습 가능하도록 설정
base_model.trainable = True

# 학습률을 낮게 설정하여 미세 조정
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 (미세 조정 단계)
model.fit(ds_train, epochs=10, validation_data=ds_test, callbacks=[lr_scheduler])

# 모델 저장
model.save('optimized_emnist_cnn_model.h5')