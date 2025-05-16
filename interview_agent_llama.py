import os
import logging
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory 
from query import retrieve_questions

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI (only once)
app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama LLM
llm = OllamaLLM(model="llama3.2")

# Initialize Memory with updated import
memory = ConversationBufferMemory()

# Define Pydantic models
class InterviewRequest(BaseModel):
    name: str
    job_role: str
    company: str
    resume: str
    job_description: str
    years_of_experience: int
    skills: List[str]


class FeedbackRequest(BaseModel):
    messages: dict
    years_of_experience: int
    skills: List[str]
    job_role: str
    custom_criteria: List[str]

# Define the root endpoint
@app.get("/")
async def root():
    return {"message": "Interview agent is running!"}

def create_dynamic_question_agent():
    template = """
    You are a highly skilled interviewer representing {company}. Your goal is to conduct a well-rounded and engaging interview with {name} for the position of {job_role}. Here are the relevant details:

    - Job Role: {job_role}
    - Job Description: {job_description}
    - Years of Experience: {years_of_experience}
    - Skills: {skills}
    - Company's Preferred Questions: {interview_qs}
    - Candidate's Resume: {resume}

    You have access to a prioritized list of interview questions: {interview_qs}.

    Your task is:

    1. Select and ask **three main interview questions** directly from the provided `interview_qs`. Do **not** generate or paraphrase — use exactly three distinct questions from the list.
    2. For **each** main question, generate **one personalized follow-up question** based on the candidate’s resume, experience, and skills. These follow-ups should:
    - Demonstrate you've reviewed the candidate's resume and experience
    - Be relevant to their real-world work, challenges, or achievements
    - Encourage reflection, depth, or elaboration
    3. Ensure all questions are clear, professional, and conversational. Address the candidate directly in the follow-ups.

    Return a JSON object in this format:

    {{
    "greeting": "A short, warm welcome to {name}, acknowledging their time and interest.",
    "questions": [
        {{
        "main_question": "<First question from interview_qs>",
        "follow_up_question": "<Follow-up tailored to the candidate's background>"
        }},
        {{
        "main_question": "<Second question from interview_qs>",
        "follow_up_question": "<Follow-up tailored to the candidate's background>"
        }},
        {{
        "main_question": "<Third question from interview_qs>",
        "follow_up_question": "<Follow-up tailored to the candidate's background>"
        }}
    ]
    }}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "name",
            "job_role",
            "company",
            "resume",
            "job_description",
            "years_of_experience",
            "skills",
            "interview_qs"
        ]
    )
    return LLMChain(llm=llm, prompt=prompt)



 
def create_feedback_analysis_agent():
    template = """
    Interview details:
    {responses}
    IMPORTANT INSTRUCTIONS:
    - Return VALID JSON with NO null values
    - Base ALL assessments on EVIDENCE from responses
    - Ensure EVERY field is populated
    - Use EXACTLY the field names provided below
    {{
    "overall_score": <integer between 0 and 100>,
    "overall_summary": "<A concise 2-3 sentence summary of the overall score>",
    "skill_analysis": {{
        "communication_skills": {{
            "clarity": <integer between 0 and 100>,
            "articulation": <integer between 0 and 100>,
            "active_listening": <integer between 0 and 100>
        }},
        "conceptual_understanding": {{
            "fundamental_concepts": <integer between 0 and 100>,
            "theoretical_application": <integer between 0 and 100>,
            "analytical_reasoning": <integer between 0 and 100>
        }},
        "speech_analysis": {{
            "avg_filler_words_used": <integer>,
            "avg_confidence_level": "<High|Medium|Low>",
            "avg_fluency_rate": <integer between 0 and 100>
        }},
        "time_management": {{
            "average_response_time": "<string, e.g., '45 seconds'>",
            "question_completion_rate": <integer between 0 and 100>
        }},
        "skills_assessment": {{
            "<each item in {skills}>": "<Score between 0 and 10>"
        }}
    }},
    "strengths_and_weaknesses": [
        <3 strengths and 3 weaknesses>
        {{
            "type": "strength",
            "title": "<title for strength>",
            "description": "<description of the strength>"
        }},
        {{
            "type": "weakness",
            "title": "<title for weakness>",
            "description": "<description of the weakness>"
        }}
    ],
    "question_analysis": [
        <for each question asked in interview>
        {{
            "question": "<question asked by bot>",
            "question_id": "<integer>",
            "transcript": "<response given by user>",
            "quick_analysis": "<quick analysis of the response>",
            "fluency_score": "<integer between 0 and 100>",
            "confidence_score": "<integer between 0 and 100>",
            "words_per_minute": "<integer>",
            "filler_words": ["<list of filler words used>"],
            "time_seconds": "<integer time it took to answer>",
            "clarity_score": "<integer between 0 and 100>",
        }}
    ],
    "performance_metrics": {{
        "critical_thinking": <integer between 0 and 100>,
        "logical_reasoning": <integer between 0 and 100>,
        "problem_solving": <integer between 0 and 100>,
        "adaptability": <integer between 0 and 100>,
        "creativity": <integer between 0 and 100>,
        "foundational_knowledge": <integer between 0 and 100>,
        "advanced_concepts": <integer between 0 and 100>,
        "practical_application": <integer between 0 and 100>,
        "articulation": <integer between 0 and 100>,
        "technical_terms": <integer between 0 and 100>,
        "active_listening": <integer between 0 and 100>
    }},
    "career_path_recommendations": [
        <3 career path recommendations>
        {{
            "recommended_role": "<role_1>",
            "skill_match": <integer between 0 and 100>,
            "skills": ["<skill1>", "<skill2>", "<skill3>"]
        }}
    ],
    "custom_metrics": {{
        "<each item in {custom_criteria}>": "<Score between 0 and 10>"
    }}
    }}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["responses", "years_of_experience", "skills", "job_role", "custom_criteria"]
    )
    return LLMChain(llm=llm, prompt=prompt)

def generate_initial_question(
    name: str,
    job_role: str,
    company: str,
    resume: str,
    job_description: str,
    years_of_experience: int,
    skills: list,
):
    # Always retrieve questions based on the job role
    interview_qs = retrieve_questions(job_role)
    print("Interview questions passed to the agent:", interview_qs)

    # Create the agent that will generate a dynamic question
    agent = create_dynamic_question_agent()

    # Invoke the agent with the full candidate and job context
    return agent.invoke(
        {
            "name": name,
            "job_role": job_role,
            "company": company,
            "resume": resume,
            "job_description": job_description,
            "years_of_experience": years_of_experience,
            "skills": skills,
            "interview_qs": interview_qs,
        }
    )

def generate_feedback(messages: dict, years_of_experience: int, skills: list, job_role: str, custom_criteria: list):
    agent = create_feedback_analysis_agent()
    return agent.invoke({
        "responses": messages,
        "years_of_experience": years_of_experience,
        "skills": skills,
        "job_role": job_role,
        "custom_criteria": custom_criteria
    })

@app.post("/generate_questions")
async def generate_questions(data: InterviewRequest):
    result = generate_initial_question(
        name=data.name,
        job_role=data.job_role,
        company=data.company,
        resume=data.resume,
        job_description=data.job_description,
        years_of_experience=data.years_of_experience,
        skills=data.skills,
    
    )
    return result

@app.post("/generate_feedback")
async def feedback_endpoint(data: FeedbackRequest):
    result = generate_feedback(
        messages=data.messages,
        years_of_experience=data.years_of_experience,
        skills=data.skills,
        job_role=data.job_role,
        custom_criteria=data.custom_criteria
    )
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)