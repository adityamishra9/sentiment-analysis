# convert_to_onnx.py
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import make_pipeline

# 1. load your vectorizer + classifier
vec, clf = joblib.load("models/sentiment_model.pkl")

# 2. combine into one pipeline
pipe = make_pipeline(vec, clf)

# 3. convert; input named "input"
onnx_model = convert_sklearn(
    pipe,
    initial_types=[("input", StringTensorType([None, 1]))]
)

# 4. save
with open("models/sentiment.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ ONNX model written to models/sentiment.onnx")
