
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle


class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
    'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque',
    'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella',
    'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine',
    'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
    'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak',
    'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
    'waffles'
]


def load_model(weights_path="/content/drive/MyDrive/Checkpoint models/best_model_resnet50.pth"):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    state_dict = torch.load(weights_path, map_location="cpu")
    

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    model.eval()
    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize(448),                
    transforms.RandomResizedCrop(384),      
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# -------------------------------------------------------------------
# 4. Prediction function
# -------------------------------------------------------------------
def predict(image, model):
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return class_names[pred.item()]

# -------------------------------------------------------------------
# 5. Streamlit UI
# -------------------------------------------------------------------
st.title("🍽️ Food-101 Classifier")
st.write("Upload a food image and the model will predict which dish it is.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Classifying...")

    pred_class = predict(image, model)
    st.success(f"### 🍛 Predicted Food Item: **{pred_class}**")
