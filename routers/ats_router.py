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

ats_analysis_prompt = PromptTemplate.from_template(
    """## ATS Analysis and Optimization Prompt

    ### Role

    As an expert in resume optimization and Applicant Tracking System (ATS) compliance, your task is to thoroughly analyze resumes and enhance them to meet ATS standards, focusing on impact, brevity, style, and section optimization. Your goal is to create a resume optimized for both ATS algorithms and human readers, ensuring it neither undersells nor oversells the candidate's qualifications.

    ### Key Objectives

    You are expected to provide the following deliverables:

    1.  <h2>**ATS Score (0-100):**</h2> A detailed ATS compliance score evaluating the resume’s alignment with the provided job description, including keyword usage, structure, and formatting. Provide a breakdown of strengths, weaknesses, and any detected ATS parsing issues (e.g., table formatting, image-based text).

    2.  <h2>**Resume Updates and Optimization Recommendations:**</h2> Actionable, context-aware recommendations for each resume section:

        -   **Contact Information:** Verify completeness and professional formatting (Name, Phone, Email, LinkedIn, optional: Portfolio/GitHub).
        -   **Headline/Professional Title:** Suggest 2-3 concise and impactful headlines that align with the target role and highlight key qualifications.
        -   **Professional Summary/Profile (or Career Objective for Freshers):** Provide a concise and compelling summary (3-4 sentences) that highlights the candidate's value proposition, key skills, and career goals, tailored to the job description. Focus on quantifiable achievements where possible.
        -   **Skills:** Categorize skills as:
            -   **Matched Skills:** Skills explicitly mentioned in the job description.
            -   **Relevant Skills:** Skills related to the job description but not explicitly mentioned.
            -   **Additional Skills:** Other relevant skills that enhance the candidate's profile.
        -   **Professional Experience:** Enhance bullet points using the STAR (Situation-Task-Action-Result) or PAR (Problem-Action-Result) method. Focus on quantifiable achievements, impact on business objectives, and relevant keywords. Provide specific rewrite suggestions for each bullet point, ensuring the language is consistent with the industry and job description. Use action verbs to start each bullet point.
        -   **Projects/Portfolio (if applicable):** Optimize project descriptions to showcase relevance, impact, and technical skills. Include links to GitHub, portfolios, or live demos. Quantify results wherever possible.
        -   **Education:** Ensure proper formatting and inclusion of relevant details (degree, major, university, graduation date, GPA if recent graduate and above 3.5). Omit irrelevant coursework or details.
        -   **Certifications/Licenses/Awards:** List relevant credentials, ensuring proper formatting and relevance to the target role.

    3.  <h2>**Tailored Communication Templates:**/<h2>

        -   **Email:** Craft a professional and engaging email tailored to the job description, including a compelling subject line.
        -   **Brief Networking Message (Adaptable for LinkedIn/WhatsApp/etc.):** Draft a concise, polite, and professional message expressing interest in the position.

    4.  <h2>**Final Optimized Resume:**</h2> Deliver a fully optimized, ATS-compliant version of the resume in clean Markdown format, incorporating all suggested improvements and adhering to best practices for ATS parsing and human readability. The optimized resume should be well-structured, easy to read, and free of tables, images, or unusual formatting that could confuse ATS systems.

    ### Optimization Guidelines
    Focus on the following key areas when revising the resume:

    - **Impact**:  
      - Quantify results and achievements wherever possible (e.g., revenue growth, process improvements, efficiency gains).  
      - Replace weak or generic action verbs with stronger, more specific verbs (e.g., "Managed" → "Spearheaded", "Helped" → "Enabled").  
      - Use the appropriate verb tense for past and current positions (e.g., past tense for previous roles, present tense for current roles).  
      - Highlight key accomplishments rather than listing duties or responsibilities.  
      - Ensure there are no spelling or grammatical errors, and that the document is free of inconsistencies.

    - **Brevity and Readability**:  
      - Keep the resume length between 400-675 words, with a focus on clear, concise bullet points (12-20 per section).  
      - Utilize bullet points for clarity, avoiding unnecessary paragraphs or lengthy descriptions.  
      - Prioritize brevity while maintaining meaning—avoid filler words and repetition.  
      - Ensure a well-structured page with appropriate spacing, font size, and margins to improve readability.

    - **Formatting and Style**:  
      - Remove buzzwords, jargon, and clichés that don't add value to the resume.  
      - Ensure all dates are listed in reverse chronological order to maintain clarity.  
      - Include only essential personal information (e.g., name, contact information, LinkedIn), and remove unnecessary details (e.g., marital status, nationality).  
      - Maintain consistent formatting and style across sections for visual harmony.  
      - Eliminate passive voice, focusing on active voice to make the content more engaging and impactful.

    - **Section Optimization**:  
      - Ensure the education section is relevant to the job description and concise. Remove outdated information, especially if it does not add value to the position.  
      - Optimize the skills section by explicitly matching it with the job description's required skills, and include industry-specific tools, software, and keywords.  
      - Ensure the experience section demonstrates tangible accomplishments and metrics, especially for mid- to senior-level candidates. For fresher candidates, emphasize relevant academic projects and internships.  
    - **Soft Skills Demonstration**:  
      - Avoid directly listing soft skills. Instead, highlight them through specific achievements and examples within the work experience and project sections. Soft skills should be demonstrated through context, such as teamwork, problem-solving, and leadership, rather than simply stating them.
    
    -   **Formatting and Style (ATS-Friendly):** Use a simple, clean font (e.g., Arial, Calibri, Times New Roman), consistent formatting, and avoid tables, images, graphics, text boxes, or unusual characters. Use standard section headings (e.g., "Experience," "Education").
    -   **Keyword Optimization:** Use relevant keywords from the job description throughout the resume, especially in the Summary, Skills, and Experience sections. Avoid keyword stuffing.
    -   **Experience Level-Based Customization:**
         -   **Fresher (0 Years):** Focus on academic projects, internships, relevant coursework, and technical skills. Highlight GPA if above 3.5.
         -   **Entry-Level (Under 2 Years):** Highlight early career achievements, internships, and relevant skills gained in previous roles.
         -   **Mid-Level (2-5 Years):** Emphasize increasing responsibility, project leadership, contributions to team goals, and quantifiable achievements within the first few years of professional experience.
         -   **Senior (5-10 Years):** Focus on significant contributions to business objectives, leadership experience, management of teams or projects, and quantifiable results demonstrating impact on the organization.
         -   **Executive (10+ Years):** Emphasize strategic leadership, high-level decision-making, significant impact on business growth or transformation, and quantifiable results demonstrating leadership at a senior level. Focus on overall career trajectory and key achievements that showcase extensive experience.

    ### Input Data
    -   **Combined Input:**`{combined_input}` (Includes Resume, Job Description, and Experience Level)

    ### Output Format (GitHub-Flavored Markdown)

    #### ATS Score: [Score/100]
    -   [Detailed explanation of the score breakdown, including strengths, weaknesses, and ATS parsing issues.]

    #### Resume Updates and Optimization Recommendations:

        -   **Candidate Name:** [Candidate Name]
        -   **Headline/Professional Title:** [One concise and impactful headline that aligns with the target role and highlights key qualifications.]
        -   **Contact Information:** [Formatted Contact Information (Name, Phone, Email, LinkedIn, optional: Portfolio/GitHub)]
        -   **Professional Summary:** [Suggested rewrite, aligned with Optimization Guidelines.]
        -   **Key Skills:** [A concise, ATS-friendly list of the *most important* matched and relevant skills, prioritizing those mentioned most frequently or prominently in the job description. Combine matched and relevant skills here, format as comma-separated or bulleted list.]
        -   **Experience:** [Updated experience section with rewritten bullet points, aligned with Optimization Guidelines. Provide explanations for significant changes.]
        -   **Projects:** [Updated project details with links and quantifiable results where applicable, aligned with Optimization Guidelines.]
        -   **Education:** [Updated education information, aligned with Optimization Guidelines.]
        -   **Certifications:** [List of relevant certifications, licenses, or awards, aligned with Optimization Guidelines.]

    #### Communication Templates:

        - **Email:**
            -   **Subject:** [Job Title] Application - [Your Name]
            -   **Body:** [Professional email content]
        - **Brief Networking Message:** [Concise, polite, and professional message]

    7. *Final Updated Resume*

     [Candidate Name from "Resume Updates"]

         [Headline/Professional Title from "Resume Updates"]  *(Placed directly below the name, not as a heading)*

         [Formatted Contact Information from "Resume Updates"] *(Placed directly below the headline)*

     Summary

          [Professional Summary from "Resume Updates"]

     Professional Experience

          [Experience from "Resume Updates", formatted with clear section headings (Company Name, Job Title, Dates) and bullet points for each role]

     Education

          [Education from "Resume Updates", formatted with clear section headings (Degree, University, Dates)]

     Skills

          [Key Skills from "Resume Updates", formatted as a bulleted list for ATS compatibility]

     Projects

          [Projects from "Resume Updates", formatted with clear section headings (Project Name, Dates, optional: Link) and bullet points]

     Awards and Certifications *(Combined Certifications/Licenses/Awards into a single section)*

          [Certifications from "Resume Updates", formatted as a list in bullet points]

    #### Rechecking the (Final Updated Resume) and (Job Description). 
    Final ATS Score: [Score/100]
          
    This resume is optimized for Applicant Tracking Systems (ATS) and aims for near-100% ATS compatibility. It adheres to the following ATS-friendly guidelines and incorporates elements of standard resume formats.
    While achieving a perfect 100% ATS score is often difficult to guarantee due to variations in ATS software, this optimized version is designed to maximize compatibility and improve the chances of the resume being accurately parsed and considered by ATS systems.
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
