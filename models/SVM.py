import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np


class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=5, batch_size=32, num_classes=3):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        Construction du modèle SVM avec TensorFlow/Keras.
        """
        model = models.Sequential()
        model.add(layers.Dense(self.num_classes,
                               activation=None,
                               kernel_regularizer=tf.keras.regularizers.l2(self.lambda_param)))
        return model

    def fit(self, training_inputs, training_labels, test_inputs=None, test_labels=None, logdir=None):
        """
        Entraînement du modèle avec validation et callbacks pour TensorBoard.
        """
        print("Training Started...")

        # Encodage des labels
        training_labels_cat = tf.keras.utils.to_categorical(training_labels, self.num_classes)
        validation_data = None
        if test_inputs is not None and test_labels is not None:
            test_labels_cat = tf.keras.utils.to_categorical(test_labels, self.num_classes)
            validation_data = (test_inputs, test_labels_cat)

        # Compilation du modèle
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalHinge(),
            metrics=['accuracy']
        )

        # Callbacks pour TensorBoard
        callbacks = []
        if logdir:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
            callbacks.append(tensorboard_callback)

        # Entraînement
        history = self.model.fit(training_inputs, training_labels_cat, validation_data=validation_data,
                                 epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks)

        print("Training Finished.")

        # Affichage des métriques finales
        self.evaluate_metrics(training_inputs, training_labels, "Training")
        if test_inputs is not None and test_labels is not None:
            self.evaluate_metrics(test_inputs, test_labels, "Test")

    def evaluate_metrics(self, inputs, labels, dataset_name=""):
        """
        Évaluation complète : calcul de la perte, F1-score et matrice de confusion.
        """
        labels_cat = tf.keras.utils.to_categorical(labels, self.num_classes)
        logits = self.model(inputs, training=False)
        loss_fn = tf.keras.losses.CategoricalHinge()
        loss = loss_fn(labels_cat, logits).numpy()

        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == labels)

        f1 = self._compute_f1(labels, predictions)
        cm = tf.math.confusion_matrix(labels, predictions)

        print(f"\n### {dataset_name} Metrics ###")  # Ajoute le nom du dataset dans l'affichage
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm.numpy())

        return loss, accuracy


    def _compute_f1(self, labels, predictions):
        """
        Calcul du F1-score à l'aide de TensorFlow/Keras.
        """
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        predictions = tf.convert_to_tensor(predictions, dtype=tf.int32)

        precision = tf.keras.metrics.Precision()
        recall = tf.keras.metrics.Recall()

        precision.update_state(labels, predictions)
        recall.update_state(labels, predictions)

        p = precision.result().numpy()
        r = recall.result().numpy()

        f1 = 2 * (p * r) / (p + r + 1e-7)  # Évite une division par zéro
        return f1


    def predict(self, inputs):
        """
        Prédiction des labels.
        """
        logits = self.model(inputs, training=False)
        return np.argmax(logits, axis=1)


    def save_model(self, filename):
        """
        Sauvegarde complète du modèle.
        """
        self.model.save(filename)
        print(f"Modèle sauvegardé sous : {filename}")


    def load_model(self, filename):
        """
        Chargement du modèle sauvegardé.
        """
        self.model = tf.keras.models.load_model(filename)
        print(f"Modèle chargé depuis : {filename}")
