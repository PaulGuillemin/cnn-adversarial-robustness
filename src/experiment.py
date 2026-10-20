"""Runnable extraction of the archived CNN/FGM/PGD experiment.

Training and attack calls retain the original notebook's behavior. File output,
CLI options and the synthetic smoke check were added for portfolio reproduction.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from tensorflow.keras import Model, layers


def build_model():
    inputs = layers.Input(shape=(32, 32, 3))
    x = inputs
    for block in range(3):
        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        if block < 2:
            x = layers.Dropout(0.30)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    return Model(inputs, layers.Dense(10)(x))


def load_data():
    import tensorflow_datasets as tfds

    def normalize(image, label):
        return tf.cast(image, tf.float32) / 127.5 - 1.0, label

    def augment(image, label):
        padded = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
        image = tf.image.random_crop(padded, tf.shape(image))
        return tf.image.random_flip_left_right(image), label

    data = tfds.load("cifar10", as_supervised=True)
    train = data["train"].map(augment).map(normalize).shuffle(10000).batch(128)
    test = data["test"].map(normalize).batch(1000)
    return train, test


def make_train_step(model):
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            # Preserve the notebook's call: no explicit training=True.
            predictions = model(x)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    return train_step


def attack_batch(model, x, epsilon, step_size, iterations):
    # No labels or pixel clipping are supplied, matching the archived calls.
    fgm = fast_gradient_method(model, x, epsilon, np.inf)
    pgd = projected_gradient_descent(model, x, epsilon, step_size, iterations, np.inf)
    return fgm, pgd


def evaluate(model, batches, epsilon, step_size, iterations):
    metrics = {name: tf.metrics.SparseCategoricalAccuracy()
               for name in ("clean", "fgm", "pgd")}
    count = 0
    for x, y in batches:
        fgm, pgd = attack_batch(model, x, epsilon, step_size, iterations)
        for name, images in (("clean", x), ("fgm", fgm), ("pgd", pgd)):
            metrics[name].update_state(y, model(images))
        count += int(tf.shape(x)[0])
        print(f"Evaluated {count} images", flush=True)
    return {"evaluated_images": count, "accuracy_percent":
            {name: float(metric.result().numpy() * 100) for name, metric in metrics.items()}}


def smoke_check():
    """Exercise a real update and both attacks without downloading CIFAR-10."""
    tf.keras.utils.set_random_seed(42)
    model = build_model()
    x = tf.random.uniform((2, 32, 32, 3), minval=-1.0, maxval=1.0)
    y = tf.constant([0, 1], dtype=tf.int64)
    before = model.trainable_variables[0].numpy().copy()
    loss = make_train_step(model)(x, y)
    tf.debugging.assert_all_finite(loss, "Non-finite training loss")
    if np.array_equal(before, model.trainable_variables[0].numpy()):
        raise AssertionError("The training step did not update the weights")
    fgm, pgd = attack_batch(model, x, 0.05, 0.01, 40)
    for name, images in (("clean", x), ("fgm", fgm), ("pgd", pgd)):
        logits = model(images)
        tf.debugging.assert_equal(tf.shape(logits), [2, 10])
        tf.debugging.assert_all_finite(logits, f"Non-finite {name} logits")
        if name != "clean":
            maximum = float(tf.reduce_max(tf.abs(images - x)).numpy())
            if not 0 < maximum <= 0.05001:
                raise AssertionError(f"Unexpected {name} perturbation: {maximum}")
            print(f"{name.upper()} Linf perturbation: {maximum:.7f}")
    print(f"PASS: training update, finite logits, FGM and 40-step PGD; {model.count_params()} parameters")
    print("Synthetic inputs and fresh weights: this is not an accuracy benchmark.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a synthetic check only")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, help="Optional seed for a new run")
    parser.add_argument("--output", type=Path, default=Path("runs/latest"))
    args = parser.parse_args()
    if args.smoke:
        smoke_check()
        return
    if args.epochs < 1:
        parser.error("--epochs must be at least one")
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    train, test = load_data()
    model = build_model()
    model.summary()
    train_step = make_train_step(model)
    for epoch in range(args.epochs):
        loss_metric = tf.metrics.Mean()
        for x, y in train:
            loss_metric.update_state(train_step(x, y))
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={float(loss_metric.result()):.5f}", flush=True)
    result = evaluate(model, test, 0.05, 0.01, 40)
    result.update({"epochs": args.epochs, "seed": args.seed,
                   "epsilon": 0.05, "pgd_step_size": 0.01, "pgd_iterations": 40,
                   "tensorflow_version": tf.__version__, "numpy_version": np.__version__})
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model.keras")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
