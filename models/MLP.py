import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class MLP:
    def __init__(self, structure: list, learning_rate: float = 0.01, epochs: int = 1000,
                 is_classification: bool = True, hidden_activation: str = 'relu') -> None:
        """
        Initialisation de l'architecture MLP utilisant TensorFlow/Keras.
        """
        self.structure = structure
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.is_classification = is_classification
        self.hidden_activation = hidden_activation
        self.model = self._build_model()

    def _build_model(self):
        """
        Construction dynamique du modèle MLP selon la structure spécifiée.
        """
        model = keras.Sequential()

        # Couches cachées
        for i, units in enumerate(self.structure[1:-1]):
            if i == 0:
                model.add(layers.Dense(units, activation=self.hidden_activation,
                                       input_shape=(self.structure[0],)))
            else:
                model.add(layers.Dense(units, activation=self.hidden_activation))

        # Couche de sortie
        output_units = self.structure[-1]
        output_activation = 'softmax' if self.is_classification else None
        model.add(layers.Dense(output_units, activation=output_activation))

        # Définir la perte
        loss = 'categorical_crossentropy' if self.is_classification else 'mse'

        # Compilation
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        metrics = ['accuracy'] if self.is_classification else []
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def fit(self, training_inputs, training_labels, test_inputs=None, test_labels=None, logdir=None):
        """
        Entraîne le modèle et affiche les métriques.
        """
        # Convertir les données en tenseurs
        training_inputs = tf.convert_to_tensor(training_inputs, dtype=tf.float32)
        training_labels = np.array(training_labels)

        # Callbacks (TensorBoard)
        callbacks = []
        if logdir:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
            callbacks.append(tensorboard_callback)

        # Conversion des labels pour classification
        if self.is_classification:
            num_classes = len(np.unique(training_labels))
            training_labels_cat = tf.keras.utils.to_categorical(training_labels, num_classes)
            test_labels_cat = tf.keras.utils.to_categorical(test_labels,
                                                            num_classes) if test_labels is not None else None
        else:
            training_labels_cat = training_labels
            test_labels_cat = test_labels

        # Entraînement
        k = self.model.fit(
            training_inputs, training_labels_cat,
            validation_data=(test_inputs, test_labels_cat) if test_inputs is not None else None,
            epochs=self.epochs, batch_size=32, callbacks=callbacks
        )

        print("TRAINING FINISHED \n")

        # Affichage des métriques d'évaluation
        print("\n### Évaluation sur l'ensemble d'entraînement ###")
        self._evaluate_metrics(training_inputs, training_labels)

        if test_inputs is not None and test_labels is not None:
            print("\n### Évaluation sur l'ensemble de test ###")
            self._evaluate_metrics(test_inputs, test_labels)

    def _evaluate_metrics(self, inputs, labels):
        """
        Évalue les métriques : F1-score et matrice de confusion.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        predictions = self.predict(inputs)

        if self.is_classification:
            # Calcul du F1-score (Keras)
            f1 = self._compute_f1(labels, predictions)
            print(f"F1-Score : {f1:.4f}")

            # Calcul de la matrice de confusion (TensorFlow)
            cm = tf.math.confusion_matrix(labels, predictions)
            print("Matrice de confusion :")
            print(cm.numpy())

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

    def predict(self, input):
        """
        Prédiction des sorties pour les données d'entrée.
        """
        input = tf.convert_to_tensor(input, dtype=tf.float32)
        predictions = self.model.predict(input, verbose=0)

        if self.is_classification:
            return np.argmax(predictions, axis=1)
        return predictions

    def save_model(self, filename: str):
        """
        Sauvegarde complète du modèle (architecture, poids, optimizer, etc.).
        """
        self.model.save(filename)
        print(f"Modèle sauvegardé sous : {filename}")

    def load_model(self, filename: str):
        """
        Chargement complet du modèle sauvegardé.
        """
        self.model = keras.models.load_model(filename)
        print(f"Modèle chargé depuis : {filename}")
