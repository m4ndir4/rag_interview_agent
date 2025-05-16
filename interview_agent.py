import os
import logging
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import OllamaLLM
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Interview agent is running!"}


# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Initialize Free OpenAI model (for general use)
llm = OllamaLLM(model="llama3")


# Initialize Paid OpenAI model (ensure you have access to GPT-4 or similar)
# paid_llm = OpenAI(
#     temperature=0.7,
#     model_name="gpt-4",
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
# )

# Initialize Memory
memory = ConversationBufferMemory()


def create_dynamic_question_agent():
    template =  """
    You are a highly skilled interviewer representing {company}. Your goal is to conduct a well-rounded and engaging interview with {name} for the position of {job_role}. Here are the relevant details:
    Job Role: {job_role}.
    Job Description: {job_description}.
    Years of Experience: {years_of_experience}.
    Skills: {skills}.
    Company's Preferred Questions: {interview_qs}.
    Candidate's Resume: {resume}

    Your task is to generate a series of 10 conversational and adaptive questions that flow naturally as part of a two-way dialogue. The interview should feel engaging, thoughtful, and well-tailored to the candidate's background and the job role. Follow these guidelines:

    1. Begin with a conversational opener that invites a story-driven introduction, e.g., 'Walk me through your journey in {job_role}—what sparked your interest, and how has your {years_of_experience} years of experience shaped your approach?'
    2. At least *three* deeply *technical* questions that test core expertise based on the candidate's resume and the skills required for job role.
    3. The remaining *six* should assess *problem-solving, frameworks/tools proficiency, real-world application, best practices, debugging, and cultural fit*.
    4. Ask atleast 2 follow up question on prevoius questions. Ask follow ups on previous questions with phrases like 'Let's dig deeper into...' or 'Based on what you shared about [specific detail], how would you...'. Embed follow-ups naturally: After technical/situational questions, add 'Walk me through your thought process there' or 'What trade-offs did you consider?'
    5. Personalize transitions: Use name in 3-4 questions (e.g., '{name}, if we collaborated on [job-related task], how would you...')
    6. Reflect {company}'s values when framing cultural fit questions.
    7. Adjust the difficulty level of questions to match the candidate's years of experience.
    8. Ask one question at a time and keep each question to the point.
    9. Incorporate the company's preferred questions naturally and contextually.
    10. Construct resume specified questions.

    Return a JSON object in the following format (with no placeholder text).
    {{
        "greeting": "Welcome the candidate by name with genuine enthusiasm, acknowledging their time and interest",
        "questions": [ <comma seperated list of questions. Do not number the questions.> ]
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
        ],
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
    interview_qs: list
):
    agent = create_dynamic_question_agent()
    return agent.invoke(
        {
            "name": name,
            "job_role": job_role,
            "company": company,
            "resume": resume,
            "job_description": job_description,
            "years_of_experience": years_of_experience,
            "skills": skills,
            "interview_qs": interview_qs
        }
    )


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
    prompt = PromptTemplate(template=template, input_variables=["responses", "years_of_experience", "skills", "job_role", "custom_criteria"])
    return LLMChain(llm=llm, prompt=prompt)


def generate_feedback(messages: dict, years_of_experience: int, skills: list, job_role: str, custom_criteria: str):
    agent = create_feedback_analysis_agent()
    return agent.invoke({
        "responses": messages,
        "years_of_experience": years_of_experience,
        "skills": skills,
        "job_role": job_role,
        "custom_criteria": custom_criteria
    })




#to test
from typing import List
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

# CORS config (optional but useful if testing from a frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request input
class InterviewRequest(BaseModel):
    name: str
    job_role: str
    company: str
    resume: str
    job_description: str
    years_of_experience: int
    skills: List[str]
    interview_qs: List[str]

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
        interview_qs=data.interview_qs
    )
    return result
