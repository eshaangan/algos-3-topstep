import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Load model
model_path = Path("ml_intraday_v3/models/saved/model_bundle_balanced_v3.pkl")
bundle = joblib.load(model_path)
model = bundle['primary_model']
features = bundle['primary_feature_columns']

# Get feature importance
importance = model.feature_importances_
feature_imp = pd.DataFrame({'feature': features, 'importance': importance})
feature_imp = feature_imp.sort_values('importance', ascending=False)

print("\nTop 20 Features:")
print(feature_imp.head(20))

print(f"\n'side' feature rank:")
print(feature_imp[feature_imp['feature'] == 'side'])
