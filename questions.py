from vectorstore_setup import get_vectorstore

def insert_questions():
    vectorstore = get_vectorstore()

    data = [
        # SOFTWARE ENGINEER
        ("Describe a time you optimized code performance.", {"job_role": "software_engineer", "question_type": "technical"}),
        ("How do you ensure code quality in a team?", {"job_role": "software_engineer", "question_type": "behavioral"}),
        ("What design patterns have you used and why?", {"job_role": "software_engineer", "question_type": "technical"}),
        ("How do you handle tight deadlines in development projects?", {"job_role": "software_engineer", "question_type": "behavioral"}),
        ("Explain your debugging process.", {"job_role": "software_engineer", "question_type": "technical"}),
        ("How do you prioritize features in a sprint?", {"job_role": "software_engineer", "question_type": "agile"}),
        ("Tell me about a challenging bug you resolved.", {"job_role": "software_engineer", "question_type": "technical"}),
        ("How do you stay updated with new technologies?", {"job_role": "software_engineer", "question_type": "behavioral"}),
        ("Describe your experience with CI/CD pipelines.", {"job_role": "software_engineer", "question_type": "technical"}),
        ("How do you handle code reviews?", {"job_role": "software_engineer", "question_type": "collaboration"}),

        # MARKETING AGENT
        ("Describe a successful ad campaign you worked on.", {"job_role": "marketing_agent", "question_type": "campaign"}),
        ("How do you measure ROI in digital marketing?", {"job_role": "marketing_agent", "question_type": "analytical"}),
        ("What strategies do you use to grow social media engagement?", {"job_role": "marketing_agent", "question_type": "strategic"}),
        ("Tell me about a time you handled a public relations crisis.", {"job_role": "marketing_agent", "question_type": "crisis"}),
        ("How do you tailor messaging to different audiences?", {"job_role": "marketing_agent", "question_type": "communication"}),
        ("What tools do you use for campaign analytics?", {"job_role": "marketing_agent", "question_type": "technical"}),
        ("Describe your experience with content calendars.", {"job_role": "marketing_agent", "question_type": "organizational"}),
        ("How do you manage collaboration with sales teams?", {"job_role": "marketing_agent", "question_type": "collaboration"}),
        ("How do you determine the success of a branding campaign?", {"job_role": "marketing_agent", "question_type": "analytical"}),
        ("Give an example of creative problem solving in marketing.", {"job_role": "marketing_agent", "question_type": "creative"}),

        # DATA ANALYST
        ("Explain a time you used data to influence a decision.", {"job_role": "data_analyst", "question_type": "analytical"}),
        ("How do you handle missing values in datasets?", {"job_role": "data_analyst", "question_type": "technical"}),
        ("What data visualization tools are you most comfortable with?", {"job_role": "data_analyst", "question_type": "technical"}),
        ("Describe a project where your analysis had a major impact.", {"job_role": "data_analyst", "question_type": "impact"}),
        ("How do you ensure data integrity and accuracy?", {"job_role": "data_analyst", "question_type": "technical"}),
        ("Walk me through how you would analyze customer churn.", {"job_role": "data_analyst", "question_type": "case_study"}),
        ("How do you present complex findings to non-technical stakeholders?", {"job_role": "data_analyst", "question_type": "communication"}),
        ("Describe your process for cleaning a messy dataset.", {"job_role": "data_analyst", "question_type": "technical"}),
        ("What’s the most challenging data problem you’ve solved?", {"job_role": "data_analyst", "question_type": "challenge"}),
        ("How do you decide which metrics to track?", {"job_role": "data_analyst", "question_type": "analytical"}),
    ]

    texts = [item[0] for item in data]
    metadatas = [item[1] for item in data]

    vectorstore.add_texts(texts, metadatas)
    vectorstore.persist()
    print("✅ Interview questions successfully added to Chroma DB.")

if __name__ == "__main__":
    insert_questions()