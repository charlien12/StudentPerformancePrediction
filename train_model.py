import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
# Load dataset
data = pd.read_csv('student.csv')
data.rename(columns={'Attendance (%)':'Attendance','Exam Score':'Score','Study Hours':'Hours'}, inplace=True)
X=data[['Attendance', 'Hours']]
y=data['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)

# Evaluate

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print(f"Fast Model RMSE: {rmse:.4f}")

print(f"Fast Model R²: {r2:.4f}")  # This will show ~0.80

# Save model

joblib.dump(model, 'student_performance_model.pkl')

print("Model saved as student_performance_model.pkl")
 