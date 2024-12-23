import os
import pandas as pd
import tensorflow as tf
import numpy as np
from models.MLP import MLP
from models.SVM import SVM

if __name__ == '__main__':
    # Charger les données CSV
    data = pd.read_csv("../data/accidents_merge.csv")

    # Sélectionner les caractéristiques et la cible
    inputs = data.drop(columns=['grav'])  # Remplacez 'grav' par la colonne cible
    labels = data['grav']

    # Encodage des labels pour classification
    labels = labels.astype('category').cat.codes  # Encode les labels catégoriels en entiers

    # Normalisation des entrées (remplace StandardScaler de sklearn)
    inputs_mean = inputs.mean(axis=0)
    inputs_std = inputs.std(axis=0)
    input_scaled = (inputs - inputs_mean) / inputs_std


    # Division des données en ensembles d'entraînement, validation et test
    def split_data(inputs, labels, test_size=0.2, seed=42):
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.shuffle(buffer_size=len(inputs), seed=seed)
        test_size = int(len(inputs) * test_size)
        train_size = len(inputs) - test_size
        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        return train_dataset, test_dataset


    # Train/test split
    train_dataset, test_dataset = split_data(input_scaled.values, labels.values)
    train_inputs, train_labels = tuple(zip(*train_dataset.as_numpy_iterator()))
    test_inputs, test_labels = tuple(zip(*test_dataset.as_numpy_iterator()))

    # Validation split
    train_dataset, val_dataset = split_data(np.array(train_inputs), np.array(train_labels), test_size=0.2)
    train_inputs, train_labels = tuple(zip(*train_dataset.as_numpy_iterator()))
    val_inputs, val_labels = tuple(zip(*val_dataset.as_numpy_iterator()))


    # Conversion en tenseurs
    def convert_to_tensor(inputs, labels):
        return tf.convert_to_tensor(inputs, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.int32)


    train_inputs, train_labels = convert_to_tensor(train_inputs, train_labels)
    test_inputs, test_labels = convert_to_tensor(test_inputs, test_labels)
    val_inputs, val_labels = convert_to_tensor(val_inputs, val_labels)

    # Nombre de classes
    num_classes = len(np.unique(labels))

    # Configuration des répertoires
    base_logdir = "../logs/"
    seeds = [20, 17, 7, 3, 28]

    for seed in seeds:
        # Fixer la seed pour garantir la reproductibilité
        tf.keras.utils.set_random_seed(seed)

        """SVM"""

        # Entraînement
        svm = SVM(learning_rate=0.001, lambda_param=0.01, num_classes=num_classes, epochs=5)
        # Chemin pour TensorBoard logs
        experiment_name = f"seed_{seed}_LR_{svm.learning_rate}_epochs_{svm.epochs}_lambda_{svm.lambda_param}"
        final_logdir = os.path.join(base_logdir, "SVM", experiment_name)

        svm.fit(train_inputs, train_labels, test_inputs, test_labels, logdir=final_logdir)

        # Sauvegarde et chargement du modèle
        svm.save_model("../model_parameters/SVM/svm_model1.keras")
        svm.load_model("../model_parameters/SVM/svm_model1.keras")

        # Prédictions
        predictions = svm.predict(val_inputs)
        print(f"FINAL PREDICTION {predictions}")

        # Évaluation sur les données de Validation
        print("\nÉvaluation finale sur le jeu de Validation :")
        svm.evaluate_metrics(val_inputs, val_labels)

        """END OF SVM"""

    exit(0)

    """MLP"""

    # Initialisation du MLP
    mlp = MLP(structure=[train_inputs.shape[1], 32, 16, num_classes], learning_rate=0.01, epochs=5,
              is_classification=True, hidden_activation="relu")

    # Chemin pour TensorBoard logs
    experiment_name = f"seed_{seed}_structure_{mlp.structure}_LR_{mlp.learning_rate}_epochs_{mlp.epochs}_hidden_activation_{mlp.hidden_activation}"
    final_logdir = os.path.join(base_logdir, "MLP", experiment_name)

    # Entraînement
    mlp.fit(train_inputs, train_labels, test_inputs=test_inputs, test_labels=test_labels, logdir=final_logdir)

    # Sauvegarde et chargement du modèle
    mlp.save_model("../model_parameters/MLP/mlp_model1.keras")
    mlp.load_model("../model_parameters/MLP/mlp_model1.keras")

    print(f"FINAL PREDICTION \n {mlp.predict(val_inputs)}")

    # Évaluation sur les données de Validation
    print("\nÉvaluation finale sur le jeu de Validation :")
    mlp._evaluate_metrics(val_inputs, val_labels)

    """END OF MLP"""