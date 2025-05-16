from vectorstore_setup import get_vectorstore

def retrieve_questions(job_role: str = None, question_type: str = None, query: str = "interview question"):
    vectorstore = get_vectorstore()

    # Build metadata filter
    filter_criteria = {}
    if job_role:
        filter_criteria["job_role"] = job_role
    if question_type:
        filter_criteria["question_type"] = question_type

    # Perform similarity search
    results = vectorstore.similarity_search(query=query, k=10, filter=filter_criteria)

    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")

if __name__ == "__main__":
    retrieve_questions(job_role="software_engineer")
