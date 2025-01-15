import unittest
import tensorflow as tf
from tensorflow.keras.models import Sequential


class TestMNISTModel(unittest.TestCase):
    
    def setUp(self):
        # Инициализация данных для тестов
        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
            tf.keras.datasets.mnist.load_data()
        # Отбор 10% данных
        fraction = 0.1
        num_train_samples = int(len(self.x_train) * fraction)
        num_test_samples = int(len(self.x_test) * fraction)

        self.x_train_10 = self.x_train[:num_train_samples].astype('float32') / 255
        self.y_train_10 = \
            tf.keras.utils.to_categorical(self.y_train[:num_train_samples], 10)
        self.x_test_10 = self.x_test[:num_test_samples].astype('float32') / 255
        self.y_test_10 = \
            tf.keras.utils.to_categorical(self.y_test[:num_test_samples], 10)

    def test_data_shapes(self):
        # Проверка размеров данных
        self.assertEqual(self.x_train_10.shape[0], int(len(self.x_train) * 0.1))
        self.assertEqual(self.x_test_10.shape[0], int(len(self.x_test) * 0.1))
        self.assertEqual(self.x_train_10.shape[1:], (28, 28))
        self.assertEqual(self.y_train_10.shape[1], 10)

    def test_model_training(self):
        # Проверка обучения модели
        model = Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(self.x_train_10, self.y_train_10,
                            epochs=1, batch_size=32, 
                            validation_data=(self.x_test_10, self.y_test_10))
        # Проверка, что после обучения хотя бы одна эпоха прошла
        self.assertGreater(len(history.history['loss']), 0)
        self.assertGreaterEqual(history.history['accuracy'][0], 0)
        self.assertGreaterEqual(history.history['val_accuracy'][0], 0)

    def test_model_evaluation(self):
        # Проверка оценки модели
        model = Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.x_train_10, self.y_train_10, epochs=1, batch_size=32)
        test_loss, test_acc = model.evaluate(self.x_test_10, self.y_test_10)
        # Проверка, что точность находится в допустимых пределах
        self.assertGreaterEqual(test_acc, 0)
        self.assertLessEqual(test_acc, 1)


if __name__ == '__main__':
    unittest.main()
