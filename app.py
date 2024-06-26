#run the app
#python -m streamlit run d:/NSFW/Project/test1.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import math, keras_ocr
# Initialize pipeline
pipeline = None
model_path="NSFW_text_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

import streamlit as st

def get_distance(predictions):
    """
    Function returns dictionary with (key,value):
        * text : detected text in image
        * center_x : center of bounding box (x)
        * center_y : center of bounding box (y)
        * distance_from_origin : hypotenuse
        * distance_y : distance between y and origin (0,0)
    """

    # Point of origin
    x0, y0 = 0, 0

    # Generate dictionary
    detections = []
    for group in predictions:

        # Get center point of bounding box
        top_left_x, top_left_y = group[1][0]
        bottom_right_x, bottom_right_y = group[1][1]
        center_x, center_y = (top_left_x + bottom_right_x)/2, (top_left_y + bottom_right_y)/2

        # Use the Pythagorean Theorem to solve for distance from origin
        distance_from_origin = math.dist([x0,y0], [center_x, center_y])

        # Calculate difference between y and origin to get unique rows
        distance_y = center_y - y0

        # Append all results
        detections.append({
                            'text': group[0],
                            'center_x': center_x,
                            'center_y': center_y,
                            'distance_from_origin': distance_from_origin,
                            'distance_y': distance_y
                        })

    return detections

def distinguish_rows(lst, thresh=10):
    """Function to help distinguish unique rows"""
    sublists = []
    for i in range(0, len(lst)-1):
        if (lst[i+1]['distance_y'] - lst[i]['distance_y'] <= thresh):
            if lst[i] not in sublists:
                sublists.append(lst[i])
            sublists.append(lst[i+1])
        else:
            yield sublists
            sublists = [lst[i+1]]
    yield sublists

# Title of the app
st.title("NSFW Content Detector")

# File uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def initialize():
    global pipeline
    if pipeline==None:
        pipeline=keras_ocr.pipeline.Pipeline()

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', width=200)
    #st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    initialize()
    # Read in image
    read_image = keras_ocr.tools.read(uploaded_file)

    # prediction_groups is a list of (word, box) tuples
    prediction_groups = pipeline.recognize([read_image])
    predictions = prediction_groups[0] # extract text list
    predictions = get_distance(predictions)
    
    # Set thresh higher for text further apart
    predictions = list(distinguish_rows(predictions, thresh=10))

    # Remove all empty rows
    predictions = list(filter(lambda x:x!=[], predictions))

    # Order text detections in human readable format
    ordered_preds = []
    for row in predictions:
        row = sorted(row, key=lambda x:x['distance_from_origin'])
        for each in row: ordered_preds.append(each['text'])

    # Join detections into sentence
    sentance = ' '.join(ordered_preds)
    #st.write(sentance)

    input_text =sentance
    print(input_text)
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.softmax(dim=-1)
    print(predictions[0][0],predictions[0][1])
    if predictions[0][0]>predictions[0][1]:
        print('safe')
        st.write('Safe for Work')
    else:
        print('Not safe')
        st.write('Not Safe for Work')
    