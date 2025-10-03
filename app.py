from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import warnings
import requests
import gradio as gr
import re
from google import genai

warnings.filterwarnings('ignore')

# Load the pretrained Vision Transformer and feature extractor
model_name = 'google/vit-base-patch16-224'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# API keys
api_key_ninja = 'E5gwnDo2MSBBxbU5JasZQA==3hGc0WlGx4CkOV69'
gemini_api_key = "AIzaSyBMi12-CkGObvJDTmxGg9p6gpRF38gn63E"

def identify_image(image_path):
    """Identify the food item in the image"""
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    food_name = predicted_label.split(',')[0]
    return food_name

def get_calories(food_name):
    """Get nutrition info from API Ninjas and calories from Gemini"""
    api_url = f'https://api.api-ninjas.com/v1/nutrition?query={food_name}'
    response = requests.get(api_url, headers={'X-Api-Key': api_key_ninja})
    if response.status_code == requests.codes.ok:
        nutrition_info = response.json()
    else:
        nutrition_info = [{'Error': response.status_code, 'Message': response.text}]
    
    # Gemini client
    client = genai.Client(api_key=gemini_api_key)

    response_gemini = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Provide the estimated total calories in numeric form only (in kcal) for the entire {food_name}. Assume it is a standard portion of typical size for that food. Do not include text or units — only give the number."
    )

    content = response_gemini.text
    match = re.search(r'\d+', content)
    calories = match.group() if match else "N/A"
    return nutrition_info, calories

def format_nutrition_table(nutrition_info, calories):
    """Format nutrition info as a health-themed HTML table with dark text"""
    if isinstance(nutrition_info, list) and len(nutrition_info) > 0:
        nutrition_info[0]['Calories'] = f"{calories} kcal"

    def prettify_column(col_name):
        col_name = col_name.replace("_", " ").title()
        col_name = col_name.replace("G", "grams").replace("Mg", "mg").replace("Kcal", "kcal")
        return col_name

    html = """
    <div style='background-color:#e8f5e9; padding:20px; border-radius:12px; width:75%;'>
        <h3 style='text-align:center; color:#2e7d32;'>Nutrition Facts</h3>
        <table style='border-collapse: collapse; width: 100%; font-family: Arial;'>
        <tr style='background-color:#2e7d32; color:white;'>
            <th style='padding:10px;'>Nutrition</th>
            <th style='padding:10px;'>Value</th>
        </tr>
    """

    if isinstance(nutrition_info, list):
        for key, value in nutrition_info[0].items():
            if value == "Only available for premium subscribers.":
                continue
            key_pretty = prettify_column(key)
            html += f"""
            <tr style='border-bottom:1px solid #ccc; color:#000;'>
                <td style='padding:8px; color:#000;'>{key_pretty}</td>
                <td style='padding:8px; color:#000;'>{value}</td>
            </tr>
            """
    html += "</table></div>"
    return html


def main_process(image_path):
    food_name = identify_image(image_path)
    nutrition_info, calories = get_calories(food_name)
    return format_nutrition_table(nutrition_info, calories)

def gradio_interface(image):
    return main_process(image)

# Gradio UI
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type='filepath', label="Upload your food image"),
    outputs=gr.HTML(label="Nutrition Information"),
    title='🍏 Food Identification & Nutrition Tracker',
    description='Upload an image of food and get a clean, health-themed nutritional breakdown.',
    theme='default',
    allow_flagging='never',
    live=False
)

if __name__ == '__main__':
    iface.launch()
