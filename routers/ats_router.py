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
    You are a highly skilled resume optimization expert specializing in Applicant Tracking System (ATS) compliance. Your role is to analyze resumes against specific job descriptions, ensuring alignment with ATS requirements, industry standards, and the most relevant keywords. The **final updated resume** must achieve a **100% ATS Score**.

    Your output should include:
    1. ATS Score out of 100.
    2. Specific Updates for each resume section.
    3. A professional Email and WhatsApp Message tailored to the job description.
    4. The final **ATS-optimized Resume**, using active voice, measurable results, and proper keyword integration with clean formatting.

    ---

    ### Instructions
    #### 1. Analyze ATS Score:
    - Compare the input resume with the provided job description.
    - Evaluate keyword match, skills relevance, and experience alignment.
    - Assign an ATS Score (out of 100), clearly indicating where improvements are needed.
    - The goal is to achieve **100% ATS compliance**.

    #### 2. Optimized Resume Updates:
    - **Professional Summary**:
      - Clear, concise, tailored to the job description.
      - Use action verbs, measurable achievements, and strategic keywords.
      - Title the section "Professional Summary."
      - Ensure this section helps in achieving the **100% ATS Score**.
    
    - **Skills Section**:
      - List **only relevant hard skills, tools, and certifications** explicitly mentioned in the job description.
      - Match skills exactly as in the JD (e.g., "Python" vs. "Python Programming").
      - Show the **Exact Category** and clearly match skills between Resume and JD.
      - Ensure all required skills are integrated to achieve **100% ATS compliance**.
    
    - **Experience Section**:
      - Rewrite experience with action verbs, measurable results, and metrics.
      - Align with keywords and tools mentioned in the JD.
      - Each bullet point should contribute to the **100% ATS Score**.

    - **Projects Section** (if applicable):
      - Highlight relevant projects.
      - Emphasize tools, technologies, and quantifiable outcomes.
      - Optimize descriptions to improve ATS keyword relevance.

    - **Education Section**:
      - Ensure alignment with JD requirements.
      - Include GPA, achievements, or certifications if applicable.
      - Tailor content to ensure the resume achieves **100% ATS compliance**.

    #### 3. Professional Email and WhatsApp Message:
    - **Email**:
      - A professional, concise email tailored to the job role.
      - Include a clear subject line and align with key qualifications.

    - **WhatsApp Message**:
      - Craft a short and polite message expressing interest and relevant experience.

    #### 4. Final ATS-Optimized Resume:
    - The final resume **must achieve 100% ATS Score**:
      - Use standard headers: **Professional Summary, Skills, Experience, Education, Projects, Certifications.**
      - Integrate measurable achievements, exact keywords, and active voice.
      - Use clean formatting:
        - Fonts: Arial, Calibri, or Times New Roman.
        - No tables, images, or graphics.
        - Bullet points for readability.

    ---

    ### Input
    - Chat History: {chat_history}
    - Combined Input: {combined_input} (Resume, Job Description, and Experience Level)

    ---

    ### Output
    1. **ATS Score:** [Score/100]
       - Confirm if the score meets the target **100% ATS Score**.
    
    2. **Specific Updates:**
       - **Professional Summary:**
         [Updated professional summary tailored to the JD. Ensure alignment for 100% ATS Score.]
       - **Skills Section:**
         - **Exact Category:** [Skill Category from JD]
         - **Exact Skills from JD:** [Matched skills from JD]
         - **Exact Skills from Resume:** [Matched skills from Resume]
         - Confirm integration for **100% ATS compliance**.
       - **Experience Section:**
         [Updated experience with action verbs, metrics, and relevant keywords aligned for ATS optimization.]
       - **Projects Section:**
         [Refined project descriptions aligned to JD to maximize ATS relevance.]
       - **Education Section:**
         [Updated education details including relevant achievements to align with ATS requirements.]

    3. **Email to Recruiter:**
       **Subject:** [Job Title] Application - [Your Name]
       
       **Body:**
       Dear [Recruiter Name],

       I am excited to apply for the [Job Title] position at [Company Name]. My expertise in [Key Skills] and success in [Relevant Experience/Project] align perfectly with the role.

       I have attached my updated resume for your review. I look forward to discussing how I can contribute to [Company Goal/Initiative].

       Best regards,  
       [Your Full Name]  
       [Phone Number] | [Email Address]
    
    4. **WhatsApp Message:**
       Hi [Recruiter Name],
       
       I’m [Your Name], and I’m excited about the [Job Title] role at [Company Name]. My background in [Key Skills] and achievements in [Relevant Experience/Project] align well with the position. Let me know if we can discuss further!
    
    5. **Final Updated ATS-Optimized Resume:**
       ---
       **[Your Full Name]**  
       [Location] | [Phone Number] | [Email] | [Portfolio/LinkedIn]
       
       **Professional Summary**  
       [Tailored summary with measurable achievements and job-specific keywords ensuring 100% ATS compliance.]
       
       **Skills**  
       - [Exact Skill from JD]  
       - [Exact Skill from Old Resume that are relavant]
       
       **Experience**  
       **[Job Title]** | [Company Name] | [Duration]  
       - [Action verb + measurable achievement + relevant keyword.]  
       - [Action verb + tools used + quantifiable impact.]
       
       **Projects**  
       **[Project Name]**  
       - [Tools used + measurable outcome.]
       - [Deployed Links]
       
       **Education**  
       [Degree] | [University] | [Year] | [GPA/Relevant Achievement]
       
       **Certifications**  
       - [Certification Name] | [Issuing Authority] | [Year]
    
    ---

    ### Experience-Level Customization
    - **Freshers**: Emphasize academic projects, internships, and skills gained during studies.
    - **Under 2 Years Experience**: Highlight early career achievements, tools, and specific JD-related skills.
    - **Over 2 Years Experience**: Focus on measurable results, leadership, and contributions to business goals.
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
