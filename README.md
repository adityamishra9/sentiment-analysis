# Sentiment Analysis — End‑to‑End (Python ➜ ONNX ➜ React‑Native/Expo)

> Train a sentiment model on product reviews in Python, export to **ONNX**, and run **on‑device** inference inside a React‑Native / Expo mobile app.

<p align="center">
  <img src="https://img.shields.io/badge/ML-pipeline-blue" />
  <img src="https://img.shields.io/badge/ONNX-export-green" />
  <img src="https://img.shields.io/badge/React--Native-Expo-black" />
</p>

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Repo Structure](#repo-structure)
- [Features](#features)
- [Quick Start](#quick-start)
  - [1) Train & Export (Python)](#1-train--export-python)
  - [2) Mobile App (React‑Native/Expo)](#2-mobile-app-react-nativeexpo)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [License](#license)

---

## Overview

This project demonstrates a **full ML product loop**:
1. **Prepare & train** a sentiment classifier on Flipkart reviews using scikit‑learn.
2. **Export** the trained pipeline (vectorizer + classifier) to **ONNX** for portable inference.
3. **Bundle** the ONNX model in a **React‑Native/Expo** app to run **offline on device**.

It’s ideal as a portfolio piece showing practical ML + mobile engineering: data cleaning, model training, format conversion, and an end‑user UI.

---

## Architecture

```
+---------------------------+
|  sentiment-logic (Python) |
|  • preprocess & label     |
|  • train sklearn model    |
|  • export ONNX            |
+-------------+-------------+
              |
              v
+---------------------------+
|  sentiment-ui (Mobile)    |
|  • React-Native / Expo    |
|  • bundles ONNX model     |
|  • on-device inference    |
+---------------------------+
```

---

## Repo Structure

```
.
├── sentiment-logic/           # Python model pipeline
│   ├── data/flipkart_data.csv
│   ├── src/
│   │   ├── preprocess.py      # tokenization + negation scope + stopword filtering
│   │   ├── features.py        # (helpers; see file)
│   │   └── model.py           # TfidfVectorizer + DecisionTreeClassifier
│   ├── run.py                 # train entrypoint (reads CSV ➜ saves .pkl)
│   ├── convert_to_onnx.py     # export the sklearn pipeline ➜ models/sentiment.onnx
│   ├── predict.py             # simple VADER baseline
│   └── models/
│       ├── sentiment_model.pkl
│       └── sentiment.onnx
└── sentiment-ui/              # React-Native/Expo app
    ├── App.js                 # simple UI and sentiment inference
    ├── index.js
    ├── metro.config.js        # adds .onnx to assetExts
    ├── assets/sentiment.onnx  # packaged ONNX model
    └── android/...            # native Android project (Gradle, manifests, etc.)
```

---

## Features

- **Negation‑aware preprocessing** (`NOT_` scope tagging for “no”, “not”, “never”, “n’t”).
- **Scikit‑learn pipeline**: `TfidfVectorizer` (1–3‑grams, 5k features) + `DecisionTreeClassifier`.
- **One‑command ONNX export** of the entire pipeline.
- **Offline mobile inference** with React‑Native/Expo (no server required).
- **Clean UI** with a single text box and result bubble.

---

## Quick Start

### Prerequisites

- **Python** ≥ 3.10
- **Node.js** ≥ 18, **Yarn** or **npm**
- **Expo CLI** (`npm i -g expo`)
- (Android) **Android SDK** / **ADB** for running on a device or emulator

> Tip: Use a virtual environment for Python and `nvm` for Node.

### 1) Train & Export (Python)

```bash
# 1) set up environment
cd sentiment-logic
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2) train on CSV (writes models/sentiment_model.pkl)
python run.py

# 3) export to ONNX (writes models/sentiment.onnx)
python convert_to_onnx.py
```

The exported ONNX is used by the mobile app (copied to `sentiment-ui/assets/sentiment.onnx`).

> Optional: Quick baseline test
```bash
python predict.py
# Enter a review when prompted; prints Positive/Negative using VADER
```

### 2) Mobile App (React‑Native/Expo)

```bash
cd ../sentiment-ui
# install deps
npm install          # or: yarn
# start Metro bundler (Expo)
npm run start        # or: yarn start

# run on Android emulator / device
npm run android      # or: yarn android

# run on web (for quick UI checks)
npm run web          # or: yarn web
```

**Note:** The app currently demonstrates sentiment with the `sentiment` npm package. With an ONNX runtime (e.g., `onnxruntime-react-native`), you can swap to **true ONNX inference**. See [Usage](#usage).

---

## Usage

### A) Current demo (JS sentiment lib)

1. Launch the app via Expo.
2. Type a review in the box.
3. Tap **Analyze** to see **Positive/Negative**.

### B) Enabling ONNX inference (advanced)

To use the exported ONNX model on device:
1. Add an ONNX runtime for RN, e.g.:
   ```bash
   npm i onnxruntime-react-native
   npx pod-install ios   # if building for iOS
   ```
2. Ensure Metro bundles `.onnx` (already configured in `metro.config.js`).
3. Load the model from `assets/sentiment.onnx` and run inference in `App.js`:
   ```js
   import { InferenceSession, Tensor } from 'onnxruntime-react-native';
   import onnxModel from './assets/sentiment.onnx';

   // example (pseudo):
   const session = await InferenceSession.create(onnxModel);
   const input = new Tensor('string', [[text]]); // model expects [["your text"]]
   const outputMap = await session.run({ input });
   // map output to label
   ```

You may need to mirror the Python preprocessing (tokenization, negation scope, stop‑word handling) in JS to get parity with training.

---

## Model Details

- **Vectorizer**: `TfidfVectorizer(max_features=5000, ngram_range=(1,3))`
- **Classifier**: `DecisionTreeClassifier(random_state=42)`
- **Labels**: `sentiment = 1 if rating >= 4 else 0` (binary positive/negative)
- **Text pipeline**: lower‑casing, **negation scope** (`NOT_` prefix for tokens following negators within a window), stop‑word removal except negations.

> ⚠️ Decision Trees are simple and fast; for production, consider `LinearSVC`/`LogReg` or transformer‑based embeddings. You can still export to ONNX.

---

## Dataset

- `sentiment-logic/data/flipkart_data.csv` — product reviews with ratings.  
- Make sure column names match the code (`review`, `rating`).  
- To use your own data, replace the CSV and keep the same schema.

---

## Roadmap

- [ ] Swap demo scorer with **onnxruntime-react-native** for true on‑device ONNX inference.
- [ ] Add confidence score and neutral class.
- [ ] Improve model (regularized linear model or small distil transformer).
- [ ] Simple REST API for remote inference alternative.
- [ ] iOS build instructions & EAS workflows readme.
- [ ] Unit tests for preprocessing parity (Python ↔︎ JS).

---

## Troubleshooting

- **Android build fails / JDK issues** → Ensure Java 17+, Android SDK installed, run `expo doctor`.
- **Metro can’t load `.onnx`** → Confirm `metro.config.js` contains `assetExts.push('onnx')`, clean cache: `expo start -c`.
- **Different predictions (Python vs app)** → Align preprocessing exactly; differences in tokenization/stop‑words change outputs.
- **EAS build signing** → Replace debug signing with your own keystore for release builds.

---

## Security Notes

- The Android project contains a **debug keystore** (`android/app/debug.keystore`) for development only.  
  **Do not** ship this to production builds—generate your own release keystore and update Gradle signing configs accordingly.
- The mobile app currently bundles the model as a static asset. If model IP is sensitive, consider server‑side inference or model encryption.
