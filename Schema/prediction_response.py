from pydantic import BaseModel, Field
from typing import Dict

class PredictionResponse(BaseModel):

    predicted_category: str = Field(..., description="Predicted insurance premium category", examples=["High"])

    confidence: float = Field(..., description="Confidence score of the prediction", examples=[0.83])
    
    class_probabilities: Dict[str, float] = Field(..., description="Probabilities for each insurance premium category", examples=[{"Low": 0.1, "Medium": 0.3, "High": 0.6}])