# Keras Spoken Digit Audio Classifier

Real deep learning audio project. It trains a Keras neural model on the Free Spoken Digit Dataset, exports the learned weights, and runs browser inference on sample WAV files or uploaded audio.

## Model

- Dataset: Free Spoken Digit Dataset
- Framework: TensorFlow / Keras
- Features: log FFT band-energy features
- Architecture: Dense(192) -> Dropout -> Dense(96) -> Dense(48) -> Dense(10 softmax)
- Test accuracy: 92.2%
- Macro F1: 92.2%
- Exported artifacts:
  - `public/model/keras_spoken_digit.keras`
  - `public/model/spoken_digit_model.json`
  - `reports/keras_spoken_digit_metrics.json`

## Train

```bash
python scripts/train_keras_spoken_digits.py
```

## Run

```bash
npm install
npm run dev
```

## Deploy

```bash
npm run build
vercel deploy --prod
```
