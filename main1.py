import streamlit as st
from CStrings.generate_cstrings import questions_to_cstring
from langchain.schema import Document
from KnowledgeBase.structure_data import Get_Knowledge_Base
# Initialize knowledge base
vs = Get_Knowledge_Base("E")
st.title("CString Generator")
st.markdown("Fill in the details below to generate characteristic strings and search the knowledge base.")
# Input fields
mood = st.text_input("Mood", "")
runtime = st.text_input("Runtime", "")
age = st.text_input("Age", "")
genre = st.text_input("Genre", "")
n_characteristics = st.number_input("Number of characteristics", min_value=1, max_value=10, value=1, step=1)
if st.button("Generate and Search"):
    user_data = {
        "mood": mood,
        "runtime": runtime,
        "age": age,
        "genre": genre
    }
    # Generate characteristic strings
    resp = questions_to_cstring(user_data, n_characteristics)
    st.subheader("Results")
    for i, char_string in enumerate(resp.prompts):
        st.markdown(f"**CString {i+1}:** {char_string.prompt}")
        similar = vs.similarity_search(char_string.prompt, k=1)
        for doc in similar:
            metadata = doc.metadata
            st.markdown("### 🎬 Most Similar Movie")
            st.markdown(f"**Title:** {metadata.get('movie_name', 'N/A')}")
            st.markdown(f"**Year:** {metadata.get('year', 'N/A')}")
            st.markdown(f"**Genre:** {metadata.get('genre', 'N/A')}")
            st.markdown(f"**Director:** {metadata.get('director', 'N/A')}")
            st.markdown(f"**Cast:** {metadata.get('cast', 'N/A')}")
            st.markdown(f"**Metascore:** {metadata.get('metascore', 'N/A')}")            
            with st.expander("📄 Full Description"):
                st.text(doc.page_content)