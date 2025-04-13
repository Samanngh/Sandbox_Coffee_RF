import pandas as pd
import numpy as np

np.random.seed(42)
num_suppliers = 100

# Define country coordinate bounding boxes
country_bounds = {
    "Brazil":      {"lat": (-33.7, 5.3),    "lon": (-73.9, -34.8)},
    "Colombia":    {"lat": (-4.2, 12.4),    "lon": (-79.0, -66.9)},
    "Ethiopia":    {"lat": (3.4, 14.9),     "lon": (33.0, 48.0)},
    "Vietnam":     {"lat": (8.2, 23.4),     "lon": (102.0, 109.5)},
    "Rwanda":      {"lat": (-2.8, -1.0),    "lon": (28.8, 30.9)},
    "Honduras":    {"lat": (12.9, 16.0),    "lon": (-89.4, -83.2)},
    "Indonesia":   {"lat": (-11.0, 6.0),    "lon": (95.0, 141.0)}
}

# Generate random supplier data
data = pd.DataFrame({
    "Supplier": [f"CoffeeSupplier_{i+1:03d}" for i in range(num_suppliers)],
    "Country": np.random.choice(list(country_bounds.keys()), num_suppliers),
    "Coffee_Quality_Score": np.clip(np.random.normal(80, 5, num_suppliers), 70, 95).round(2),
    "ESG_Score": np.clip(np.random.normal(7.5, 1.2, num_suppliers), 4, 10).round(2),
    "Certification_Fairtrade": np.random.binomial(1, 0.6, num_suppliers),
    "Certification_Organic": np.random.binomial(1, 0.4, num_suppliers),
    "Distance_km": np.random.randint(5000, 15000, num_suppliers),
    "Emissions_kg_CO2": np.clip(np.random.normal(500, 120, num_suppliers), 200, 800).round(),
    "Cost_USD": np.random.randint(8000, 25000, num_suppliers),
    "Historical_ESG_Violations": np.random.poisson(0.5, num_suppliers)
})

# Assign latitude and longitude based on country boundaries
def assign_coordinates(country):
    bounds = country_bounds.get(country)
    if bounds:
        lat = np.random.uniform(*bounds['lat'])
        lon = np.random.uniform(*bounds['lon'])
        return pd.Series([lat, lon])
    else:
        return pd.Series([np.nan, np.nan])

data[['Latitude', 'Longitude']] = data['Country'].apply(assign_coordinates)

# Create synthetic sustainability risk target
# Add a small random factor to simulate real-world noise
random_noise = np.random.rand(num_suppliers)

conditions = [
    # Mostly 'Low', but a few outliers
    ((data['ESG_Score'] >= 8) & (data['Historical_ESG_Violations'] == 0)) | (random_noise > 0.95),
    
    # Mostly 'Medium', but with fuzz
    ((data['ESG_Score'] >= 6) & (data['ESG_Score'] < 8)) | (random_noise > 0.85),
    
    # Mostly 'High', with slight randomness
    ((data['ESG_Score'] < 6) | (data['Historical_ESG_Violations'] >= 2)) | (random_noise > 0.75)
]

choices = ['Low', 'Medium', 'High']

# Assign labels
data['Sustainability_Risk'] = np.select(conditions, choices, default='Medium')


# Save to CSV
data.to_csv("coffee_suppliers_with_coords.csv", index=False)
print("Dataset with coordinates saved as 'coffee_suppliers_with_coords.csv'")

# Preview
print(data.head())
