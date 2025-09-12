import pickle
import pandas as pd

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

#Mlflow model version
MODEL_VERSION = "1.0.0"

#Get class labels from model (Important for matching probablities to class names)
class_labels = model.classes_.tolist()
def predict_output(user_input: dict):
    input_df = pd.DataFrame([user_input])
    output = model.predict(input_df)[0]

    #Get probablilities for all classes
    probablities = model.predict_proba(input_df)[0]
    confidence = max(probablities)

    # Create mapping: {class_name: probability }
    class_probs = dict(zip(class_labels, map(lambda p: round(p,4), probablities)))
    return {
        "predicted_category": output,
        "confidence": round(confidence,4),
        "class_probabilities": class_probs
    }