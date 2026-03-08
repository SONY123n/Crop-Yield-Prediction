from taipy.gui import Gui
import numpy as np
import joblib

# Load trained model and preprocesser
# Ensure these files are in the same folder as app.py
model = joblib.load("crop_model.pkl")
preprocesser = joblib.load("preprocesser.pkl")

# Default input values (Types should match your notebook training)
Crop = "Arecanut"
Crop_Year = 1997
Season = "Whole Year"
State = "Assam"
Area = 73814.0
Production = 56708.0
Annual_Rainfall = 2051.0
Fertilizer = 7024878.0
Pesticide = 22882.0

prediction_result = ""

# Prediction function
def predict(state):
    try:
        # Convert inputs to the correct numeric types as used in crop_prediction.ipynb
        features = np.array([[
            state.Crop,
            int(state.Crop_Year),
            state.Season,
            state.State,
            float(state.Area),
            float(state.Production),
            float(state.Annual_Rainfall),
            float(state.Fertilizer),
            float(state.Pesticide)
        ]])

        # Transform using the saved preprocessor
        transformed = preprocesser.transform(features)

        # Predict using the DecisionTreeRegressor
        result = model.predict(transformed)

        state.prediction_result = f"Predicted Yield: {round(result[0], 2)}"
    except Exception as e:
        state.prediction_result = f"Error: {str(e)}"

# GUI Layout
page = """
# 🌾 Crop Yield Prediction Dashboard

### Crop
<|{Crop}|input|>

### Crop Year
<|{Crop_Year}|input|>

### Season
<|{Season}|input|>

### State
<|{State}|input|>

### Area
<|{Area}|input|>

### Production
<|{Production}|input|>

### Annual Rainfall
<|{Annual_Rainfall}|input|>

### Fertilizer
<|{Fertilizer}|input|>

### Pesticide
<|{Pesticide}|input|>

<|Predict Yield|button|on_action=predict|>

## <|{prediction_result}|text|>
"""

if __name__ == "__main__":
    Gui(page=page).run(port=5001)