"""Treinamento da RNA Keras conforme arquitetura do TCC.

Exige TensorFlow, opcional para quem deseja executar só o Monte Carlo.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .core import FEATURES


def train_model(dataset: str | Path, output_dir: str | Path, epochs: int = 40, seed: int = 42) -> dict:
    try:
        import tensorflow as tf
        import joblib
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.class_weight import compute_class_weight
        from sklearn.metrics import accuracy_score, log_loss
    except ImportError as exc:
        raise RuntimeError("Instale requirements-ml.txt (TensorFlow, scikit-learn e joblib)") from exc
    tf.keras.utils.set_random_seed(seed)
    df = pd.read_csv(dataset, parse_dates=["date"]).sort_values("date", kind="stable").dropna(subset=FEATURES + ["Resultado"])
    if len(df) < 100 or df.Resultado.nunique() != 3:
        raise ValueError("Base insuficiente: necessários 100+ jogos e as três classes")
    cutoff = int(len(df) * .90)
    # Particionar por DATA, não no meio de um dia de jogos.
    cutoff_date = df.iloc[cutoff].date
    test_mask = df.date >= cutoff_date
    train = df.loc[~test_mask]
    test = df.loc[test_mask]
    val_cutoff = int(len(train) * .90)
    val_date = train.iloc[val_cutoff].date
    fit = train.loc[train.date < val_date]
    val = train.loc[train.date >= val_date]
    if min(len(fit), len(val), len(test)) == 0:
        raise ValueError("Divisão temporal deixou conjunto vazio")
    if len(set(fit.Resultado)) != 3:
        raise ValueError("Todas as três classes devem existir no treino")
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit[FEATURES])  # Nunca ajustar scaler no teste/validação.
    x_val = scaler.transform(val[FEATURES]); x_test = scaler.transform(test[FEATURES])
    y_fit = fit.Resultado.to_numpy(dtype=int)
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y_fit)
    weights = {i: float(w) for i, w in enumerate(weights)}
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_fit, y_fit, validation_data=(x_val, val.Resultado.to_numpy(dtype=int)),
              class_weight=weights, epochs=epochs, batch_size=64,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                                          restore_best_weights=True)], verbose=2)
    probs = model.predict(x_test, verbose=0)
    target = test.Resultado.to_numpy(dtype=int)
    metrics = {"treino": len(fit), "validacao": len(val), "teste": len(test),
               "data_inicio_teste": str(test.date.min().date()),
               "acuracia_teste": float(accuracy_score(target, probs.argmax(axis=1))),
               "log_loss_teste": float(log_loss(target, probs, labels=[0, 1, 2])),
               "nota": "Class weights podem reduzir a calibração das probabilidades; avaliar antes de interpretar como frequências reais."}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    model.save(output / "modelo_tcc_warner.keras")
    joblib.dump(scaler, output / "scaler_tcc_warner.joblib")
    (output / "metricas_teste.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics