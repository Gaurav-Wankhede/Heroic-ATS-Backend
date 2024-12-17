from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferMemory
from models.pdf_extractor import extract_text_from_pdf
from dotenv import load_dotenv
import os

# Initialize Router
router = APIRouter()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key is missing. Set it in the .env file as GOOGLE_API_KEY.")

# Initialize Gemini Flash LLM
gemini_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY)

# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

from langchain.prompts import PromptTemplate

ats_analysis_prompt = PromptTemplate.from_template(
    """
    ### System
    You are a highly skilled resume optimization expert specializing in Applicant Tracking System (ATS) compliance. Your role is to analyze resumes against specific job descriptions, ensuring alignment with ATS requirements, industry standards, and the most relevant keywords. The optimized resume must achieve a 100% ATS score.

    Your output should include:
    1. ATS Score out of 100.
    2. Specific Updates for each resume section.
    3. A professional Email and WhatsApp Message tailored to the job description.
    4. The final Updated Resume that is ATS-compliant, using active voice, measurable results, and proper keyword integration.

    ---

    ### Instructions
    1. Analyze ATS Score:
    - Compare the input resume with the provided job description.
    - Evaluate the keyword match, skills relevance, and experience alignment.
    - Assign an ATS Score (out of 100), clearly indicating where improvements are needed.

    2. Optimized Resume Updates:
    - Provide section-specific updates:
        - Professional Summary: Clear, concise summary tailored to the job description. Use active verbs, measurable achievements, and strategic keywords. Title the section "Professional Summary".
        - Skills Section: List only relevant hard skills, tools, and certifications explicitly mentioned in the job description. Ensure skills are listed exactly as they appear in the job description (e.g., "Python" vs. "Python Programming"). Avoid redundant or irrelevant skills.
          - Show the **Exact Category** of skills and **Exact Skills** from both the **Resume** and **Job Description (JD)**. Identify and match the skills between the two and display them with clear distinction.
        - Experience Section: Rewrite descriptions to include action verbs, metrics, and results directly related to the JD. Focus on quantifiable achievements and industry-specific keywords.
        - Projects Section (if applicable): Highlight only relevant projects aligned with the JD. Emphasize tools, technologies, and outcomes.
        - Education Section: Ensure alignment with JD requirements. Include GPA or achievements if applicable.

    3. Professional Email and WhatsApp Message:
    - Craft a professional Email with a clear subject line, tailored to the job role, expressing interest and relevant qualifications.
    - Provide a concise WhatsApp message to initiate communication with the recruiter.

    4. Final ATS-Friendly Resume:
    - Provide the final updated resume in a clean and ATS-optimized format:
        - Use standard headers: Professional Summary, Skills, Experience, Education, Projects, etc.
        - Stick to standard fonts (Arial, Calibri, Times New Roman), bullet points, and proper formatting for ATS readability.
        - Integrate measurable results, relevant keywords, and action verbs across all sections.
        - Avoid including images, tables, graphics, or irrelevant content.

    ---

    ### Input
    - Chat History: {chat_history}
    - Combined Input: {combined_input} (Resume, Job Description, and Experience Level)

    ---

    ### Output
    1. **ATS Score:** [Score/100]
    2. **Specific Updates:**
       - Professional Summary: [Updated summary text]
       - Skills Section: 
           - Exact Category: [Exact Category from JD]
           - Exact Skills from JD: [List of matched skills exactly as in JD]
           - Exact Skills from Resume: [List of matched skills exactly as in the Resume]
       - Experience Section: [Updated experience bullet points with measurable results]
       - Projects Section: [Refined project descriptions]
       - Education Section: [Updated education details]
    3. **Email to Recruiter:**
       Subject: [Job Title] Application - [Your Name]

       Dear [Recruiter Name],

       I hope this email finds you well.

       I am writing to express my keen interest in the [Job Title] position at [Company Name]. My expertise in [Key Skills] and proven success in [Relevant Experience/Project] strongly align with your team's needs. 

       Attached is my updated resume for your review. I look forward to discussing how my skills can contribute to [Company Goal or Initiative].  

       Thank you for your time and consideration.  

       Best regards,  
       [Your Full Name]  
       [Contact Information]

    4. **WhatsApp Message:**
       Hi [Recruiter Name],

       I’m [Your Name], excited about the [Job Title] role at [Company Name]. My background in [Key Skills/Tools] and achievements in [Relevant Experience/Project] align well with the position. I’d love to discuss how I can contribute to the team.

       Looking forward to your response.

    5. **Updated Resume:**
       [Your Name]  
       [Location] | [Phone Number] | [Email] | Portfolio | LinkedIn | GitHub

       Professional Summary  
       [Tailored summary with keywords, achievements, and measurable results.]

       Skills  
       - [Exact Skill from JD]
       - [Exact Skill from Resume]

       Experience  
       [Job Title] | [Company Name] | [Duration]  
       - [Action verb + quantifiable achievement + relevant keyword.]  
       - [Action verb + measurable result + tools used.]

       Projects  
       [Project Name]  
       - [Tools/Technologies used + measurable impact.]

       Education  
       [Degree] | [University Name] | [Year] | [GPA (if applicable)]

       Certifications  
       - [Certification Name] | [Issuing Authority] | [Year]

    ---

    ### Experience Level Based Customization:
    
    For Freshers:
    - Professional Summary: Focus on enthusiasm, relevant academic skills, and readiness to apply knowledge.
    - Skills Section: Highlight academic projects, internships, and tools learned during studies, matching the exact wording of skills mentioned in the job description.
    - Experience Section: Emphasize academic achievements, internships, and relevant skills.
    - Projects Section: Highlight academic projects or personal projects relevant to the job description.

    For Candidates with Under 2 Years of Experience:
    - Professional Summary: Focus on relevant early career experiences and achievements.
    - Skills Section: Include industry-specific tools, software, and skills acquired on the job, matching the terminology used in the JD.
    - Experience Section: Highlight job-related accomplishments, metrics, and specific tools used.
    - Projects Section: Mention relevant projects, including results or outcomes achieved during employment or internships.

    For Candidates with Over 2 Years of Experience:
    - Professional Summary: Emphasize leadership, measurable achievements, and specific contributions to business goals.
    - Skills Section: Focus on technical skills, certifications, and leadership competencies, exactly matching the JD's wording.
    - Experience Section: Showcase impactful results, team leadership, and business process improvements.
    - Projects Section: Highlight complex projects that had significant business impact, using quantifiable data.
    """
)



# Health Check
@router.get("/")
async def health_check():
    return {"message":"Heroic ATS Platform is Working Fine"}


# ATS Analysis Endpoint
@router.post("/analyze_ats")
async def analyze_ats(pdf_file: UploadFile = File(...), job_description: str = Form(...), experience_level: str = Form(...)):
    """
    Endpoint to analyze resume ATS compatibility against a job description.
    
    - **pdf_file**: Uploaded resume in PDF format.
    - **job_description**: Job description as input.
    - **experience_level**: Experience level of the candidate (Fresher, 2 Years of Experience, More than 2 Years).
    """
    try:
        # Validate uploaded file type
        if not pdf_file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
        # Extract text from PDF
        resume_text = await extract_text_from_pdf(pdf_file)

        # Determine experience level and incorporate into combined_input
        experience = "Fresher" if experience_level.lower() == "fresher" else "2 Years of Experience" if experience_level.lower() == "2 years" else "More than 2 Years of Experience"

        # Combine resume text, job description, and experience level into one input string
        combined_input = f"""
        Resume: {resume_text}
        
        Job Description: {job_description}
        
        Experience Level: {experience}
        """

        # Define and run the ATS analysis chain
        ats_analysis_chain = LLMChain(
            llm=gemini_flash,
            prompt=ats_analysis_prompt,
            memory=memory,
            output_parser=StrOutputParser()
        )

        # Invoke the chain asynchronously
        ats_analysis_result = await ats_analysis_chain.ainvoke({"combined_input": combined_input})
        return {"analysis_result": ats_analysis_result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Clear Memory Endpoint
@router.post("/clear_memory")
async def clear_memory():
    """
    Endpoint to clear the conversation memory.
    """
    memory.clear()
    return {"message": "Memory cleared successfully."}
