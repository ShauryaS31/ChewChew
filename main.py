# how long u should eat 
# how many chew u have to take in 1 bite
#nutrition info
#good rating for the food

from dotenv import load_dotenv
import google.generativeai as genai
import os
import base64
import time
from PIL import Image
import streamlit as st
import re
import cv2
import tempfile

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_fridge_image(image_path):
    try:
        image = Image.open(image_path)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")

        prompt_items = """
        Analyze this image of food items and provide detailed information in the following exact format:
        
        Item: [Item name]
        Quantity: [Estimated quantity or 'Not visible' if you can't determine]
        Location: [Bounding box coordinates as [ymin, xmin, ymax, xmax] where each coordinate is an integer between 0 and 1000]
        Eating Duration: [Recommended time to properly eat this food item]
        Chews Per Bite: [Recommended number of chews per bite based on food texture]
        Nutrition (per serving):
        - Calories: [estimated calories]
        - Protein: [grams]
        - Fats: [grams]
        - Carbs: [grams]
        Health Rating: [Rate from 1-10, where 10 is extremely healthy. Provide brief explanation]

        Notes for analysis:
        - For eating duration and chews:
          * Soft foods need fewer chews (e.g., yogurt: 5-10 chews)
          * Medium foods need moderate chews (e.g., cooked vegetables: 15-20 chews)
          * Hard/fibrous foods need more chews (e.g., meat, raw vegetables: 30-40 chews)
        
        - For health rating, consider:
          * Nutrient density
          * Processing level
          * Sugar/salt content
          * Fiber content
          * Overall health benefits/drawbacks
        """

        for attempt in range(3):
            try:
                response_items = model.generate_content([prompt_items, image])
                break
            except Exception as e:
                if '429' in str(e):
                    if attempt < 2:
                        st.warning("Quota exceeded. Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        st.error("Quota exceeded. Please try again later.")
                        return None, None, None
                else:
                    st.error(f"An error occurred: {str(e)}")
                    return None, None, None
        
        items_info = parse_item_info(response_items.text.strip())
        annotated_image = generate_annotated_image(image_path, items_info)

        analysis_result = ""
        for item, info in items_info.items():
            analysis_result += f"\n{item}:\n"
            analysis_result += f"  Quantity: {info['quantity']}\n"
            analysis_result += f"  Eating Duration: {info.get('eating_duration', 'Not specified')}\n"
            analysis_result += f"  Chews Per Bite: {info.get('chews_per_bite', 'Not specified')}\n"
        
        analysis_result += f"\n\nTotal number of distinct food items: {len(items_info)}"
        
        return analysis_result, annotated_image, items_info

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None, None
    finally:
        image.close()

def parse_item_info(response_text):
    items_info = {}
    current_item = None
    for line in response_text.split('\n'):
        if line.startswith("Item:"):
            current_item = line.split(":", 1)[1].strip()
            items_info[current_item] = {}
        elif line.startswith("Quantity:") and current_item:
            items_info[current_item]['quantity'] = line.split(":", 1)[1].strip()
        elif line.startswith("Location:") and current_item:
            location = line.split(":", 1)[1].strip()
            box = parse_bounding_box(location)
            items_info[current_item]['box'] = box
            items_info[current_item]['location'] = location if box == [0, 0, 0, 0] else f"Coordinates: {box}"
        elif line.startswith("Eating Duration:") and current_item:
            items_info[current_item]['eating_duration'] = line.split(":", 1)[1].strip()
        elif line.startswith("Chews Per Bite:") and current_item:
            items_info[current_item]['chews_per_bite'] = line.split(":", 1)[1].strip()
    return items_info

def parse_bounding_box(text):
    match = re.search(r'\[?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*\]?', text)
    if match:
        return [int(match.group(i)) for i in range(1, 5)]
    return [0, 0, 0, 0]

def convert_coordinates(box, original_width, original_height):
    ymin, xmin, ymax, xmax = box
    return [
        int(ymin / 1000 * original_height),
        int(xmin / 1000 * original_width),
        int(ymax / 1000 * original_height),
        int(xmax / 1000 * original_width)
    ]

def generate_annotated_image(image_path, items_info):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    for item, info in items_info.items():
        box = info['box']
        quantity = info['quantity']

        if box != [0, 0, 0, 0]:
            ymin, xmin, ymax, xmax = convert_coordinates(box, width, height)
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width - 1))
            ymax = max(0, min(ymax, height - 1))

            if xmin < xmax and ymin < ymax:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"{item}: {quantity}"
                cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def main():
    st.title("Food Analysis Assistant 🍽️")
    st.write("Upload a photo of your food or fridge contents to get eating recommendations!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display loading message
        with st.spinner('Analyzing your food...'):
            # Create a temporary file to store the uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(uploaded_file, use_column_width=True)

            # Analyze the image
            analysis_result, annotated_image, items_info = analyze_fridge_image(tmp_path)

            # Display annotated image
            with col2:
                st.subheader("Detected Items")
                if annotated_image:
                    st.image(annotated_image, use_column_width=True)

            # Display results in a nice format
            if analysis_result:
                st.subheader("Food Analysis Results")
                
                # Create expandable sections for each food item
                for item, info in items_info.items():
                    with st.expander(f"📌 {item}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Quantity:**", info['quantity'])
                            st.write("**Eating Duration:**", info.get('eating_duration', 'Not specified'))
                            st.write("**Chews Per Bite:**", info.get('chews_per_bite', 'Not specified'))
                            st.write("**Health Rating:**", info.get('health_rating', 'Not specified'))
                        
                        with col2:
                            st.write("**Nutritional Information (per serving):**")
                            if 'nutrition' in info:
                                nutrition = info['nutrition']
                                st.write(f"- Calories: {nutrition.get('Calories', 'N/A')}")
                                st.write(f"- Protein: {nutrition.get('Protein', 'N/A')}")
                                st.write(f"- Fats: {nutrition.get('Fats', 'N/A')}")
                                st.write(f"- Carbs: {nutrition.get('Carbs', 'N/A')}")

                # Display total items
                st.info(f"Total number of food items detected: {len(items_info)}")

            # Cleanup temporary file
            os.unlink(tmp_path)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Food Analysis Assistant",
        page_icon="🍽️",
        layout="wide"
    )
    main()
