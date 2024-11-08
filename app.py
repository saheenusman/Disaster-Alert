import streamlit as st
import requests
import folium
import osmnx as ox
from streamlit_folium import st_folium
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch

# Set up the API endpoint and your API key
api_url = "https://api.vultrinference.com/v1/chat/completions"
api_key = "BZINS2TPSF4KBTBSGAM47C7OZJM7TUP7A6JA"

# Function to get a response from the API
def get_chat_response(user_message):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama2-7b-chat-Q5_K_M",  # Replace with the model ID you selected
        "messages": [{"role": "user", "content": user_message}]
    }

    response = requests.post(api_url, headers=headers, json=data)
    response_json = response.json()

    return response_json['choices'][0]['message']['content']

# Set Streamlit page configuration
st.set_page_config(page_title="Kerala Disaster Alert System", layout="wide")

# Cache district boundaries for Kerala
@st.cache_data
def load_kerala_district_boundaries():
    place_name = "Kerala, India"
    tags = {'boundary': 'administrative', 'admin_level': '6'}
    gdf = ox.geometries_from_place(place_name, tags)

    # Check if 'name' exists; if not, create a fallback
    if 'name' not in gdf.columns:
        gdf['name'] = gdf.index.astype(str)  # Use the index as a fallback

    # Ensure we only keep rows with valid geometries
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    
    return gdf

# Load boundaries and retrieve district names
boundaries = load_kerala_district_boundaries()
district_names = boundaries['name'].tolist()

# Define the GCN model
class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        return h

# Load pre-trained model
model = GCNModel(in_channels=5, hidden_channels=16, num_classes=1)
model.load_state_dict(torch.load("gcn_model.pth"))
model.eval()

# Sample data for predictions
sample_node_features = torch.tensor([
    [29.0, 85.0, 15.0, 5.0, 50.0],   # Thiruvananthapuram - High Risk
    [21.0, 40.0, 2.0, 30.0, 1000.0], # Kollam - Low Risk
    [30.0, 90.0, 20.0, 8.0, 200.0],  # Pathanamthitta - High Risk
    [20.0, 35.0, 1.0, 40.0, 1200.0], # Alappuzha - Low Risk
    [28.0, 80.0, 18.0, 10.0, 300.0], # Kottayam - High Risk
    [19.0, 45.0, 3.0, 25.0, 900.0],  # Idukki - Low Risk
    [25.0, 50.0, 15.0, 15.0, 150.0], # Ernakulam - High Risk
    [22.0, 60.0, 10.0, 20.0, 400.0], # Thrissur - Low Risk
    [18.0, 55.0, 5.0, 35.0, 500.0],  # Palakkad - Low Risk
    [26.0, 65.0, 12.0, 12.0, 350.0], # Malappuram - Low Risk
    [27.0, 70.0, 14.0, 22.0, 450.0], # Kozhikode - High Risk
    [24.0, 75.0, 8.0, 30.0, 600.0],  # Wayanad - Low Risk
    [23.0, 65.0, 9.0, 27.0, 500.0],  # Kannur - Low Risk
    [29.0, 85.0, 15.0, 5.0, 50.0],   # Kasaragod - High Risk
], dtype=torch.float)

# Define edge index
sample_edge_index = torch.tensor([
    [0, 1], [0, 2], [0, 4], [1, 3], [2, 5], [3, 6],
    [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12],
    [10, 13], [11, 12]
], dtype=torch.long).t()

sample_edge_features = torch.ones((sample_edge_index.shape[1], 2), dtype=torch.float)
sample_data = Data(x=sample_node_features, edge_index=sample_edge_index, edge_attr=sample_edge_features)

# Run model inference and get predictions
with torch.no_grad():
    predictions_tensor = model(sample_data)
    predicted_probabilities = torch.sigmoid(predictions_tensor)
    predicted_labels = (predicted_probabilities > 0.5).float()

# Map district names to predictions
district_names = ["Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha", "Kottayam", 
                  "Idukki", "Ernakulam", "Thrissur", "Palakkad", "Malappuram", 
                  "Kozhikode", "Wayanad", "Kannur", "Kasaragod"]

predictions = {
    district: {"probability": prob.item(), "risk": "High Risk" if label.item() == 1 else "Low Risk"}
    for district, prob, label in zip(district_names, predicted_probabilities, predicted_labels)
}

# Render map with high-risk areas highlighted
def render_map(predictions):
    kerala_location = [10.8505, 76.2711]
    m = folium.Map(location=kerala_location, zoom_start=7)

    for _, row in boundaries.iterrows():
        district_name = row['name']
        risk_info = predictions.get(district_name, {"probability": 0.0, "risk": "Low Risk"})
        
        if risk_info["risk"] == "High Risk":
            fill_color = "red" 
            tooltip_text = f"<strong>{district_name}</strong><br>Risk Level: {risk_info['risk']} ({risk_info['probability']:.2%})"
            
            # Add district boundary to map with style based on high-risk status
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x, fill_color=fill_color: {
                    'fillColor': fill_color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.5
                },
                tooltip=tooltip_text
            ).add_to(m)

    return m

# Display map with current predictions
map_obj = render_map(predictions)
st_folium(map_obj, width=1000, key="kerala_map")

# Display high-risk regions and probabilities in text format on the sidebar
st.sidebar.header("High-Risk Districts")
for district, info in predictions.items():
    if info["risk"] == "High Risk":
        st.sidebar.write(f"**{district}** - {info['probability']:.2%}")

# Chatbot section
st.sidebar.header("Chatbot - Disaster Alert System")
user_input = st.sidebar.text_input("Ask me about the weather or disaster alerts:")

if user_input:
    chatbot_response = get_chat_response(user_input)
    st.sidebar.text_area("Chatbot:", chatbot_response, height=150)
