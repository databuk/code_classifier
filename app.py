import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
model_path = "ibukunirinyenikan/malicious-code-detector"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def predict_code(code):
    inputs = tokenizer(code, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        label_map = {0:'Clean', 1:'Malicious'}
        predicted_label = label_map[predictions.item()]
        return predicted_label
st.title("Malicious code detector")
st.write("Upload a .py script and check whether it's malicious or clean.")
uploaded_file = st.file_uploader("Upload a .py file", type=["py"])
if uploaded_file is not None:
  code = uploaded_file.read().decode("utf-8")
  if st.checkbox('Display uploaded code'):
    st.code(code, language='python')
  if st.spinner("Analyse"):
    prediction = predict_code(code)
    st.success(f"The code is {prediction}")