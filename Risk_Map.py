import pandas as pd
import folium

# Load dataset with coordinates + predicted risk
data = pd.read_csv("coffee_suppliers_with_coords.csv")

# Define color by risk level
def risk_color(risk):
    return {
        "Low": "green",
        "Medium": "orange",
        "High": "red"
    }.get(risk, "gray")

# Initialize world map
m = folium.Map(location=[0, 0], zoom_start=2)

# Add suppliers to the map
for _, row in data.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=7,
        color=risk_color(row["Sustainability_Risk"]),
        fill=True,
        fill_opacity=0.85,
        popup=f"{row['Supplier']} ({row['Country']})\nRisk: {row['Sustainability_Risk']}"
    ).add_to(m)

# Save and export
m.save("coffee_supplier_risk_map.html")
print("🌍 Map saved as: coffee_supplier_risk_map.html")
