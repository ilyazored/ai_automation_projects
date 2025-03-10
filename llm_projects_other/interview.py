import os
import PyPDF2
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
import google.generativeai as genai
import speech_recognition as sr
import time
from gtts import gTTS
import tempfile

# Set page config at the very beginning
st.set_page_config(page_title="AI-Powered Job Application Assistant", layout="wide")

# Set up API keys
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize tools
search_tool = DuckDuckGoSearchRun()

# Initialize Gemini model
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Define agents
def initialize_agents():
    researcher = Agent(
        role="Job Market Researcher",
        goal="Research the company, industry trends, and job requirements",
        backstory="You are an expert at gathering and analyzing information about companies and job markets.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=gemini
    )

    profiler = Agent(
        role="Candidate Profiler",
        goal="Analyze the candidate's resume and extract key information",
        backstory="You specialize in understanding candidate profiles and matching them to job requirements.",
        verbose=True,
        allow_delegation=False,
        llm=gemini
    )

    resume_strategist = Agent(
        role="Resume Strategist for Engineers",
        goal="Find the best ways to make a resume stand out in the job market",
        verbose=True,
        backstory=(
            "With a strategic mind and an eye for detail, you "
            "excel at refining resumes to highlight the most "
            "relevant skills and experiences, ensuring they "
            "resonate perfectly with the job's requirements."
        ),
        llm=gemini
    )

    interview_preparer = Agent(
        role="Interview Preparation Specialist",
        goal="Prepare candidates for interviews by generating relevant questions and strategies",
        backstory="You are an expert at preparing candidates for job interviews, with deep knowledge of common and technical interview questions.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=gemini
    )

    return researcher, profiler, resume_strategist, interview_preparer

# Function to read PDF resume
def read_pdf_resume(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Define tasks
def create_tasks(researcher, profiler, resume_strategist, interview_preparer, resume_text, job_description, company_name):
    research_task = Task(
        description=f"Research the company {company_name} and the job market for the following position:\n{job_description}\nProvide insights on company culture, industry trends, and key requirements for the role.",
        expected_output="A detailed report on the company, industry trends, and job requirements.",
        agent=researcher
    )

    profile_task = Task(
        description=f"Analyze the following resume and extract key information about the candidate:\n{resume_text}\nProvide a summary of the candidate's skills, experience, and how they align with the job description.",
        context = [research_task],
        expected_output ="A comprehensive profile of the candidate based on their resume.",
        agent=profiler
    )

    resume_strategy_task = Task(
        description=f"Based on the job description and the candidate's profile, provide strategic advice on how to improve the resume:Offer specific, actionable recommendations to enhance the resume for this position.",
        context = [research_task,profile_task],
        expected_output="A list of strategic recommendations to improve the resume for the specific job.",
        agent=resume_strategist
    )

    interview_preparation_task = Task(
        description=f"Prepare a set of potential interview questions and strategies based on the job description and candidate's profile:\nJob Description: {job_description}\nCandidate Profile: [To be filled by Profiler]\nProvide a mix of general and technical questions, along with preparation tips.",
        context = [research_task,profile_task,resume_strategy_task],
        expected_output="A set of interview questions and preparation strategies tailored to the job and candidate.",
        agent=interview_preparer
    )

    return research_task, profile_task, resume_strategy_task, interview_preparation_task

# Function to run the job application crew
def resume_improvement_crew(resume_text, job_description, company_name):
    researcher, profiler, resume_strategist, interview_preparer = initialize_agents()
    
    research_task, profile_task, resume_strategy_task, interview_preparation_task = create_tasks(
        researcher, profiler, resume_strategist, interview_preparer, resume_text, job_description, company_name
    )

    job_application_crew = Crew(
        agents=[researcher, profiler, resume_strategist],
        tasks=[research_task, profile_task,resume_strategy_task],
        verbose=True
    )

    result = job_application_crew.kickoff()
    return result

def genrate_questions_crew(resume_text, job_description, company_name):
    researcher, profiler, resume_strategist, interview_preparer = initialize_agents()
    
    research_task, profile_task, resume_strategy_task, interview_preparation_task = create_tasks(
        researcher, profiler, resume_strategist, interview_preparer, resume_text, job_description, company_name
    )

    job_application_crew = Crew(
        agents=[profiler, resume_strategist, interview_preparer],
        tasks=[profile_task, resume_strategy_task, interview_preparation_task],
        verbose=True
    )

    result = job_application_crew.kickoff()
    return result

# Streamlit app
def main():
    st.title("AI-Powered Job Application Assistant")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    job_description = st.text_area("Enter the job description you're applying for:")
    company_name = st.text_input("Enter the company name:")
    
    if uploaded_file is not None and job_description and company_name:
        resume_text = read_pdf_resume(uploaded_file)
        
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your application and preparing recommendations..."):
                result = genrate_questions_crew(resume_text, job_description, company_name)
            
            st.subheader("Interview preparation recommendations:")
            st.write(result)
        if st.button("resume Improvments"):
            with st.spinner("Analyzing your application and preparing recommendations..."):
                result = resume_improvement_crew(resume_text, job_description, company_name)
            
            st.subheader("Resume Improvments:")
            st.write(result)

if __name__ == "__main__":
    main()