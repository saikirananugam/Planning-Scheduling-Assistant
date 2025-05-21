# This Python script `train_default_model.py` is performing the following tasks:
# train_default_model.py

import pickle
from src.ingest import DataIngestor
from src.model import DelayPredictor  # Adjust if needed

# 1. Load and clean data using your DataIngestor
data_path = 'data/sales_data.csv'
ingestor = DataIngestor(data_path)
df = ingestor.load_and_clean()

# 2. Define your selected default features
default_features = [
    'Zone', 'Region Zone', 'Top Level Branch', 'Top Level Type', 'Line Amount', 'SRP1',
    'Last Next Status', 'SC Rep', 'lead_time', 'is_urgent', 'total_units'
]

# 3. Define hyperparameters for the default model
hyperparams = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    # Add any others if you wish
}
model_type = 'Random Forest'

# 4. Instantiate and train the model
predictor = DelayPredictor(
    features=default_features,
    model_type=model_type,
    hyperparams=hyperparams,
    use_gridsearch=False,
    gridsearch_params={},
    train_size=0.8
)
predictor.fit(df)

# 5. Save the trained predictor as a pickle file
with open('default_predictor.pkl', 'wb') as f:
    pickle.dump(predictor, f)

print(" Default predictor trained and saved as default_predictor.pkl")
