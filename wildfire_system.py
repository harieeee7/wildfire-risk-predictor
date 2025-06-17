import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class WildfireDataSystem:
    def __init__(self, data_dir="wildfire_data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.features = [
            'latitude', 'longitude', 'temperature', 'humidity',
            'precipitation', 'wind_speed', 'landcover_type',
            'month', 'day', 'elevation', 'ndvi'
        ]
        self.model = None

    def ensure_datasets(self):
        fire_path = os.path.join(self.data_dir, "fire_data_cleaned.csv")
        weather_path = os.path.join(self.data_dir, "weather_data.csv")
        landcover_path = os.path.join(self.data_dir, "land_cover.csv")

        if not os.path.exists(fire_path):
            raise FileNotFoundError(f"{fire_path} not found! Please clean and place your fire data there.")

        if os.path.exists(weather_path) and os.path.exists(landcover_path):
            print("✅ Using existing weather and land cover data.")
            return

        print("🔄 Generating weather and land cover data...")
        fire_data = pd.read_csv(fire_path)

        # ✅ Use 'acq_date' robustly as date source
        if 'acq_date' in fire_data.columns:
            fire_data['date'] = pd.to_datetime(fire_data['acq_date'], errors='coerce')
        else:
            raise KeyError("Your fire_data must have 'acq_date' to generate 'date'.")

        fire_data = fire_data.dropna(subset=['date'])

        dates = pd.date_range(fire_data['date'].min(), fire_data['date'].max())
        weather_data = pd.DataFrame({
            'date': dates,
            'temperature': 15 + 15 * np.sin(2*np.pi*dates.dayofyear/365) + np.random.normal(0, 3, len(dates)),
            'humidity': np.clip(50 + 30 * np.sin(2*np.pi*dates.dayofyear/365) + np.random.normal(0, 10, len(dates)), 20, 80),
            'precipitation': np.clip(np.random.exponential(0.2, len(dates)), 0, 10),
            'wind_speed': np.clip(np.random.weibull(1.5, len(dates))*10, 0, 20)
        })
        weather_data.to_csv(weather_path, index=False)

        locations = fire_data[['latitude', 'longitude']].drop_duplicates()
        land_cover = pd.DataFrame({
            'latitude': locations['latitude'],
            'longitude': locations['longitude'],
            'landcover_type': np.random.choice([10, 20, 30, 40, 50, 60], len(locations)),
            'elevation': np.random.uniform(100, 1000, len(locations))
        })
        land_cover.to_csv(landcover_path, index=False)

        print("✅ Weather and land cover data generated.")

    def load_and_preprocess(self):
        """Merge fire, weather, land cover and synthesize non-fire samples if needed."""
        fire_data = pd.read_csv(os.path.join(self.data_dir, "fire_data_cleaned.csv"))
        weather_data = pd.read_csv(os.path.join(self.data_dir, "weather_data.csv"))
        land_cover = pd.read_csv(os.path.join(self.data_dir, "land_cover.csv"))

        # ✅ Consistent: build 'date' only from 'acq_date'
        if 'acq_date' in fire_data.columns:
            fire_data['date'] = pd.to_datetime(fire_data['acq_date'], errors='coerce')
        else:
            raise KeyError("Your fire_data must have 'acq_date' to generate 'date'.")

        fire_data = fire_data.dropna(subset=['date'])
        weather_data['date'] = pd.to_datetime(weather_data['date'])

        fire_data['month'] = fire_data['date'].dt.month
        fire_data['day'] = fire_data['date'].dt.day

        merged = pd.merge_asof(
            fire_data.sort_values('date'),
            weather_data.sort_values('date'),
            on='date',
            direction='nearest'
        )

        from scipy.spatial import KDTree
        tree = KDTree(land_cover[['latitude', 'longitude']])
        _, idx = tree.query(merged[['latitude', 'longitude']])
        merged['landcover_type'] = land_cover.iloc[idx]['landcover_type'].values
        merged['elevation'] = land_cover.iloc[idx]['elevation'].values
        merged['ndvi'] = merged['landcover_type'].apply(
            lambda x: 0.8 if x in [30, 40] else 0.6 if x in [20, 60] else 0.3 if x == 50 else 0.1
        )
        merged['fire_occurred'] = (merged['frp'] > 0).astype(int)

        if merged['fire_occurred'].nunique() == 1:
            print("⚠️ Detected only fire samples. Generating synthetic non-fire samples...")
            non_fire_samples = merged.sample(n=len(merged)//2, random_state=42).copy()
            non_fire_samples['temperature'] -= np.random.uniform(5, 15, len(non_fire_samples))
            non_fire_samples['humidity'] = np.clip(non_fire_samples['humidity'] + np.random.uniform(20, 40, len(non_fire_samples)), 40, 100)
            non_fire_samples['precipitation'] += np.random.uniform(1, 5, len(non_fire_samples))
            non_fire_samples['wind_speed'] = np.clip(non_fire_samples['wind_speed'] - np.random.uniform(2, 5, len(non_fire_samples)), 0, 20)
            non_fire_samples['frp'] = 0
            non_fire_samples['fire_occurred'] = 0
            merged = pd.concat([merged, non_fire_samples], ignore_index=True)

        X = merged[self.features]
        y = merged['fire_occurred']

        return X, y

    def train(self):
        print("🚀 Training model using local cleaned fire data...")
        self.ensure_datasets()
        X, y = self.load_and_preprocess()

        if len(np.unique(y)) < 2:
            print("⚠️ Only one class found in data. Adding synthetic opposite class to ensure training works.")
            synthetic_opposite = X.iloc[0:1].copy()
            synthetic_opposite[:] = X.mean()
            X = pd.concat([X, synthetic_opposite], ignore_index=True)
            y = pd.concat([y, pd.Series([1 - y.iloc[0]])], ignore_index=True)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight='balanced', random_state=42, n_jobs=-1
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("✅ Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        self.model = pipeline
        self.save_model()
        return pipeline

    def predict_risk(self, user_input: dict):
        if not self.model:
            raise ValueError("Model not trained yet. Please train first.")
        input_df = pd.DataFrame([{
            f: user_input.get(f, 0) for f in self.features
        }])
        proba = self.model.predict_proba(input_df)[0]
        probability = float(proba[1]) if len(proba) > 1 else (float(proba[0]) if self.model.classes_[0] == 1 else 0.0)
        return {
            'probability': probability,
            'risk_level': (
                "Extreme" if probability > 0.9 else
                "Very High" if probability > 0.7 else
                "High" if probability > 0.5 else
                "Moderate" if probability > 0.3 else
                "Low"
            ),
            'features': user_input
        }

    def save_model(self, path="wildfire_model.pkl"):
        joblib.dump(self.model, path)

    def load_model(self, path="wildfire_model.pkl"):
        self.model = joblib.load(path)
        return self.model

if __name__ == "__main__":
    system = WildfireDataSystem()
    if not os.path.exists("wildfire_model.pkl"):
        system.train()
    else:
        system.load_model()

    user_input = {
        'latitude': 34.5,
        'longitude': -118.5,
        'temperature': float(input("Enter temperature (°C): ")),
        'humidity': float(input("Enter humidity (%): ")),
        'precipitation': float(input("Enter precipitation (mm): ")),
        'wind_speed': float(input("Enter wind speed (km/h): ")),
        'landcover_type': int(input("Enter landcover type (10,20,30,...): ")),
        'month': datetime.now().month,
        'day': datetime.now().day,
        'elevation': float(input("Enter elevation (m): ")),
        'ndvi': float(input("Enter NDVI (0 to 1): "))
    }

    result = system.predict_risk(user_input)
    print("\n🌲 Wildfire Risk Prediction:")
    print(f"Probability: {result['probability']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
