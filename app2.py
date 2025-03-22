import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to Rank Resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

def generate_matching_comments(score, job_description):
    if score > 80:
        return "Excellent match! Your resume aligns very well with the job description. Consider highlighting specific achievements that relate to the job."
    elif score > 60:
        return "Good match! Your resume covers many relevant skills, but you might want to tailor it further by including keywords from the job description."
    elif score > 40:
        return "Fair match. While your resume has some relevant experience, consider revising it to better reflect the job requirements and include more specific examples."
    else:
        return "Needs improvement. Your resume does not align well with the job description. Focus on including relevant skills and experiences that match the job."

# Main execution block to run the Streamlit app
if __name__ == "__main__":
    st.title("AI Resume Screening & Ranking System")
    
    # Sidebar for user inputs
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    job_description = st.text_area("Enter Job Description")

    if uploaded_files and job_description:
        resumes = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)

        # Rank resumes
        scores = rank_resumes(job_description, resumes)

        # Display results
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        # Generate comments for each resume
        comments = [generate_matching_comments(score, job_description) for score in scores]

        # Display results with comments
        for i, row in results.iterrows():
            st.write(f"Resume: {row['Resume']}, Score: {row['Score']}%, Comment: {comments[i]}")

        # Show best match and bar chart visualization
        top_resume = results.iloc[0]
        st.success(f"Best Match: {top_resume['Resume']} with {top_resume['Score']}% match!")
        st.bar_chart(results.set_index("Resume")["Score"])
        
        # Store previous scores for comparison
        if 'previous_scores' not in st.session_state:
            st.session_state.previous_scores = []
        st.session_state.previous_scores.append(scores)

        # Compare with previous scores
        if len(st.session_state.previous_scores) > 1:
            previous_scores = st.session_state.previous_scores[-2]
            comparison_df = pd.DataFrame({
                "Current Score": scores,
                "Previous Score": previous_scores
            }, index=[file.name for file in uploaded_files])
            st.write("Comparison with Previous Scores:")
            st.bar_chart(comparison_df)
