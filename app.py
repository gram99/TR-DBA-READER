import streamlit as st
import pandas as pd
from transformers import pipeline
import io

# --- Page Configuration ---
st.set_page_config(page_title="NSPIRE Appeal Analyzer", layout="wide")

# --- UI Header ---
st.title("🛡️ NSPIRE Appeal Reason Classifier")
st.markdown("""
Upload your inspection export to automatically classify **'Non-Existent Deficiencies'**.
The system will identify the core reason for the successful appeal and flag low-confidence results.
""")

# --- 1. Load AI Model (Cached for Performance) ---
@st.cache_resource
def load_classifier():
    # Using the same high-performing model optimized for your Mac/Server
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

# --- 2. File Template & Uploader ---
st.sidebar.header("Data Input")
template_data = "Deficiency_ID|Inspectable_Area|NSPIRE_Standards|Appeal_Reason|Appeal_Comments|Mitigation_Details"
st.sidebar.download_button("📥 Download Template", data=template_data, file_name="template.txt")

uploaded_file = st.file_uploader("Upload Inspection Export (.txt or .csv)", type=['txt', 'csv'])

if uploaded_file:
    # Load and initial cleaning
    df = pd.read_csv(uploaded_file, sep="|")
    
    # Filter for Non-Existent (Handling hidden characters)
    df['Appeal_Reason'] = df['Appeal_Reason'].astype(str).str.replace(r'[^\x00-\x7F]+', '-', regex=True)
    non_existent = df[df['Appeal_Reason'].str.contains("Non-existent", na=False)].copy()

    if non_existent.empty:
        st.warning("No 'Non-existent deficiency' records found in the uploaded file.")
    else:
        st.info(f"Found {len(non_existent)} successful appeals to analyze.")

        if st.button("🚀 Run AI Analysis"):
            with st.spinner("Analyzing text and calculating confidence..."):
                # Combine text for context and clean for Excel compatibility
                non_existent['text_for_ai'] = (
                    non_existent['Appeal_Comments'].fillna('') + " " + 
                    non_existent['Mitigation_Details'].fillna('')
                ).str.replace('"', '')

                # 3. Optimized Label Strategy (Consolidated to boost confidence scores)
                labels = [
                    "Standard Misinterpretation (Incorrectly Cited)", 
                    "Exemption/Code Compliance (Safety Design or Grandfathered)", 
                    "Evidence Conflict (Photo/Fact Mismatch)", 
                    "Non-Reportable Item (Low Voltage or Resident-Owned)"
                ]

                # Run analysis
                results = classifier(
                    non_existent['text_for_ai'].tolist(), 
                    candidate_labels=labels, 
                    multi_label=False # Focus on the primary reason
                )

                # Extract top labels and scores
                non_existent['AI_Reason'] = [res['labels'][0] for res in results]
                non_existent['Confidence'] = [round(res['scores'][0], 2) for res in results]
                
                # 4. Add Review Flag (If confidence < 0.60)
                non_existent['Human_Review_Needed'] = non_existent['Confidence'].apply(
                    lambda x: "🚩 YES" if x < 0.60 else "No"
                )

                # --- Display Results ---
                st.success("Analysis Complete!")
                
                # Summary View
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Analyzed", len(non_existent))
                with col2:
                    flags = (non_existent['Human_Review_Needed'] == "🚩 YES").sum()
                    st.metric("Flagged for Review", flags)

                st.subheader("Analysis Preview")
                st.dataframe(non_existent[['Deficiency_ID', 'AI_Reason', 'Confidence', 'Human_Review_Needed']])

                # --- 5. Export to Excel ---
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Summary Sheet
                    non_existent.to_excel(writer, index=False, sheet_name='Detailed Analysis')
                    
                    # Create a quick pivot table sheet
                    pivot = pd.crosstab(non_existent['Inspectable_Area'], non_existent['AI_Reason'])
                    pivot.to_excel(writer, sheet_name='Summary by Area')

                st.download_button(
                    label="💾 Download Final Excel Report",
                    data=output.getvalue(),
                    file_name="NSPIRE_AI_Audit_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
