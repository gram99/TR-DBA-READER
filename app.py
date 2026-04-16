import streamlit as st
import pandas as pd
from transformers import pipeline
import io
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="NSPIRE Appeal Analyzer", layout="wide")

# --- UI Header ---
st.title("🛡️ NSPIRE Appeal Reason Classifier")
st.markdown("""
Upload your inspection export to automatically classify **'Non-Existent Deficiencies'**.
This dashboard identifies which standards and areas are prone to inspector error.
""")

# --- 1. Load AI Model (Cached) ---
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

# --- 2. Sidebar & File Uploader ---
st.sidebar.header("Data Input")
template_data = "Deficiency_ID|Inspectable_Area|NSPIRE_Standards|Appeal_Reason|Appeal_Comments|Mitigation_Details"
st.sidebar.download_button("📥 Download Template", data=template_data, file_name="template.txt")

uploaded_file = st.file_uploader("Upload Inspection Export (.txt or .csv)", type=['txt', 'csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="|")
    
    # Filter for Non-Existent
    df['Appeal_Reason'] = df['Appeal_Reason'].astype(str).str.replace(r'[^\x00-\x7F]+', '-', regex=True)
    non_existent = df[df['Appeal_Reason'].str.contains("Non-existent", na=False)].copy()

    if non_existent.empty:
        st.warning("No 'Non-existent deficiency' records found.")
    else:
        st.info(f"Found {len(non_existent)} successful appeals to analyze.")

        if st.button("🚀 Run AI Analysis"):
            with st.spinner("Generating High-Level Analytics..."):
                # Prepare text context
                non_existent['text_for_ai'] = (
                    non_existent['Appeal_Comments'].fillna('') + " " + 
                    non_existent['Mitigation_Details'].fillna('')
                ).str.replace('"', '')

                # 3. AI Labels
                labels = [
                    "Standard Misinterpretation (Incorrectly Cited)", 
                    "Exemption/Code Compliance (Safety Design or Grandfathered)", 
                    "Evidence Conflict (Photo/Fact Mismatch)", 
                    "Non-Reportable Item (Low Voltage or Resident-Owned)"
                ]

                # Run classification
                results = classifier(non_existent['text_for_ai'].tolist(), candidate_labels=labels)

                # Extract the top label and the top score specifically
                non_existent['AI_Reason'] = [res['labels'][0] if isinstance(res['labels'], list) else res['labels'] for res in results]
                non_existent['Confidence'] = [round(res['scores'][0] if isinstance(res['scores'], list) else res['scores'], 2) for res in results]
                                                
                # Flag low confidence (< 0.60)
                non_existent['Human_Review_Needed'] = non_existent['Confidence'].apply(
                    lambda x: "🚩 YES" if x < 0.60 else "No"
                )

                # --- High Level Metrics ---
                st.success("Analysis Complete!")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Total Records", len(non_existent))
                with m2:
                    flags = (non_existent['Human_Review_Needed'] == "🚩 YES").sum()
                    st.metric("Flagged for Review", flags)
                with m3:
                    top_std = non_existent['NSPIRE_Standards'].mode()[0]
                    st.metric("Top Problem Standard", top_std)

                # --- 4. Side-by-Side Charts ---
                st.divider()
                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("Distribution of Reasons")
                    fig_pie = px.pie(
                        non_existent, 
                        names='AI_Reason', 
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    st.subheader("Top 5 Problem Standards")
                    top_5_data = non_existent['NSPIRE_Standards'].value_counts().nlargest(5).reset_index()
                    top_5_data.columns = ['Standard', 'Count']
                    fig_bar = px.bar(
                        top_5_data, 
                        x='Count', 
                        y='Standard', 
                        orientation='h',
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)

                # --- 5. Table & Export ---
                st.subheader("Detailed Preview")
                st.dataframe(non_existent[['Deficiency_ID', 'NSPIRE_Standards', 'AI_Reason', 'Confidence', 'Human_Review_Needed']], use_container_width=True)

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    non_existent.to_excel(writer, index=False, sheet_name='Detailed Analysis')
                    # Pivot for Standard by Reason
                    std_pivot = pd.crosstab(non_existent['NSPIRE_Standards'], non_existent['AI_Reason'])
                    std_pivot.to_excel(writer, sheet_name='Summary by Standard')

                st.download_button(
                    label="💾 Download Final Excel Report",
                    data=output.getvalue(),
                    file_name="NSPIRE_AI_Audit_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
