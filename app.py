import os
import numpy as np
import joblib
from taipy.gui import Gui

# 1. Load trained model and preprocesser
# Make sure these are in your GitHub repo!
model = joblib.load("crop_model.pkl")
preprocesser = joblib.load("preprocesser.pkl")

# 2. Initial state variables
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

# 3. Prediction logic
def predict(state):
    try:
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

        transformed = preprocesser.transform(features)
        result = model.predict(transformed)
        state.prediction_result = f"Predicted Yield: {round(result[0], 4)} t/ha"
    
    except Exception as e:
        state.prediction_result = f"Error: {str(e)}"

# 4. Horizontal GUI Layout
# The 'columns=1 1' creates two equal columns
page = """
# 🌾 Crop Yield Prediction Dashboard

<|layout|columns=1 1|
<|
### 📍 Location & Time
**Crop Type**
<|{Crop}|input|>

**Year**
<|{Crop_Year}|input|>

**Season**
<|{Season}|input|>

**State**
<|{State}|input|>
|>

<|
### 🚜 Farm Statistics
**Area (Hectares)**
<|{Area}|input|>

**Production (Tonnes)**
<|{Production}|input|>

**Annual Rainfall (mm)**
<|{Annual_Rainfall}|input|>

**Fertilizer (kg)**
<|{Fertilizer}|input|>

**Pesticide (kg)**
<|{Pesticide}|input|>
|>
|>

<|Predict Yield|button|on_action=predict|>

---
## <|{prediction_result}|text|>
"""

# 5. Render Deployment Fix
if __name__ == "__main__":
    # Use the port Render gives us
    port = int(os.environ.get("PORT", 10000))
    # Bind to 0.0.0.0 for public access
    Gui(page=page).run(host="0.0.0.0", port=port)