import streamlit as st
from CStrings.iterative import iterative_cstring_gen 
from langchain.schema import Document
from KnowledgeBase.structure_data import Get_Knowledge_Base

vs = Get_Knowledge_Base("E")
st.title("CString Generator")
st.markdown("Fill in the details below to generate characteristic strings and search the knowledge base.")

# Input fields
mood = st.text_input("Mood", "")
runtime = st.text_input("Runtime", "")
age = st.text_input("Age", "")
genre = st.text_input("Genre", "")
n_characteristics = st.number_input(
    "Number of characteristics", min_value=1, max_value=10, value=1, step=1
)

if st.button("Generate and Search"):
    user_data = {
        "mood": mood,
        "runtime": runtime,
        "age": age,
        "genre": genre,
    }
    # Generate characteristic strings
    resp = iterative_cstring_gen(user_data, 3, 5)
    st.subheader("Results")
    for idx, iterres in enumerate(resp):
        # st.write(type(iterres))
        st.markdown(f"### Preference {idx + 1}:")
        similar_docs = vs.similarity_search(iterres.prompt, k = 1)
        if similar_docs:
            for doc in similar_docs:
                metadata = doc.metadata
                st.markdown("#### 🎬 Most Similar Movie")
                st.markdown(f"**Title:** {metadata.get('movie_name', 'N/A')}")
                st.markdown(f"**Year:** {metadata.get('year', 'N/A')}")
                st.markdown(f"**Genre:** {metadata.get('genre', 'N/A')}")
                st.markdown(f"**Director:** {metadata.get('director', 'N/A')}")
                st.markdown(f"**Cast:** {metadata.get('cast', 'N/A')}")
                st.markdown(f"**Metascore:** {metadata.get('metascore', 'N/A')}")
                with st.expander("📄 Full Description"):
                    st.text(doc.page_content)
        else:
            st.info("No similar movies found for this preference.")
        st.markdown("---")
