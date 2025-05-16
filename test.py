from interview_agent_gpt import generate_initial_question

data = {
    "name": "Alice",
    "job_role": "marketing_agent",
    "company": "TechCorp",
    "resume": "Experienced in Python, FastAPI, and machine learning.",
    "job_description": "Develop APIs and ML models.",
    "years_of_experience": 5,
    "skills": ["Python", "FastAPI", "ML"],
    "interview_qs": ["What is your experience with REST APIs?", "Explain your ML projects."]
}


# Only call once, with the correct arguments
result = generate_initial_question(
    name=data["name"],
    job_role=data["job_role"],
    company=data["company"],
    resume=data["resume"],
    job_description=data["job_description"],
    years_of_experience=data["years_of_experience"],
    skills=data["skills"],
    
)

print(result)
