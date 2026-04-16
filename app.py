import streamlit as st
import pandas as pd
from transformers import pipeline
import io

# 1. Title and Template Download
st.title("NSPIRE Appeal Reason Classifier")
st.markdown("Upload your inspection export to automatically classify 'Non-Existent Deficiencies'.")

# Provide a template for users to download
template_data = "Deficiency_ID|Inspectable_Area|NSPIRE_Standards|Appeal_Reason|Appeal_Comments|Mitigation_Details"
st.download_button("Download Data Template", data=template_data, file_name="template.txt")

# 2. Load AI Model (Cached to prevent re-loading on every click)
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

# 3. File Upload
uploaded_file = st.file_uploader("Upload your .txt or .csv file", type=['txt', 'csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="|")
    
    # Filter for Non-Existent
    df['Appeal_Reason'] = df['Appeal_Reason'].str.replace(r'[^\x00-\x7F]+', '-', regex=True)
    non_existent = df[df['Appeal_Reason'].str.contains("Non-existent", na=False)].copy()
    
    if st.button("Run AI Analysis"):
        with st.spinner("Classifying reasons..."):
            # Combine text and analyze
            non_existent['text_for_ai'] = non_existent['Appeal_Comments'].fillna('') + " " + non_existent['Mitigation_Details'].fillna('')
            
            labels = ["NSPIRE Standard Misinterpretation", "Safety Design Exemption", "Insufficient Photo Evidence", "Low Voltage System", "Grandfathered Item", "Lead-Free Area"]
            
            results = classifier(non_existent['text_for_ai'].tolist(), candidate_labels=labels)
            
            # Extract results
            non_existent['AI_Reason'] = [res['labels'][0] for res in results]
            non_existent['Confidence'] = [round(res['scores'][0], 2) for res in results]

            # 4. Display Results on Screen
            st.success("Analysis Complete!")
            st.dataframe(non_existent[['Deficiency_ID', 'AI_Reason', 'Confidence']])
            
            # 5. Export to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                non_existent.to_excel(writer, index=False, sheet_name='Analysis')
            
            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
                file_name="AI_Inspection_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
