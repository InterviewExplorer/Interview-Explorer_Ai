import openai
from dotenv import load_dotenv
import pdfplumber
import os
import asyncio

async def pdf(pdf_path, max_retries=3):
    # Load environment variables
    load_dotenv()

    # Set up OpenAI API key
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY is missing.")
    
    openai.api_key = api_key

    # Extract text from PDF
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
            
    prompt = f"""
    # Role
    Perform the task of extracting information from a resume received in PDF format.

    # Instructions
    Extract information from the following content:
    - PDF content:{text}

    # Task1 : Name
    - Only write the name in Korean without any spaces.

    # Task2 : Date of birth
    - Ensure that the date of birth strictly follows the '0000-00-00' format.
    - If only the age is provided, calculate the year based on the age, but do not infer the month or day.
    - If there is no information at all about the date of birth, indicate it as '없음'.

    # Task3 : Number of projects
    - The number of projects must be expressed only in numerical form.

    # Task4 : Project description
    - From the resume content, summarize the purpose of each project in a few words, based on the project name or description provided in the project section.

    # Task5 : Work experience
    - Sum up all the periods of work experience mentioned in the resume and express the total only in years and months. When the duration is indicated as 'present', 'currently employed', etc., calculate the period up to the current date.
    - Work experience is only recognized if it includes the company name.
    - If there is no information about work experience, indicate it as '없음'.
    - In expressing work experience, omit the unit (years or months) if its value is 0, and if the total period is 0, indicate it as '없음'.

    # Task6 : Technical skills
    - Based on the resume content, extract the technical skills possessed by the interviewee in a "key":"value" format. In this case, the "key" represents the category of the skill, and the "value" represents the name of the skill.
    - Do not extract information about technical skills as keywords.

    # Task7 : Summary keywords
    - Extract a minimum of 1 and a maximum of 5 keywords based on the resume content.
    - Compose the keywords only with words that represent the interviewee's personal aspects in terms of character.
    - Do not include words related to technical skills or job categories in the keywords. For example, '개발자' or 'AI'.
    - Add '#' before each keyword and separate them with spaces.

    # Policy
    - Never add any other special characters under any circumstances.
    - Use only the content that is present in the resume. Do not arbitrarily infer or add any information that is not in the resume.
    - Write all information in Korean, except for the technical skills.
    - Return the results according to the 'Example Format'.

    # Example Format:
    "name": "홍길동"
    "date_of_birth": "2024-10-02"
    "number_of_projects": "3개"
    "project_description": "쇼핑몰, 교육플랫폼, 블로그"
    "work_experience": "1년 3개월"
    "technical_skills": "백엔드: Spring Boot, Node.js, Django / 프론트엔드: React, Angular, Vue.js / ai: TensorFlow / 도구: Git / db: PostgreSQL, MongoDB/ 머신러닝: Keras, scikit-learn / 언어: java, python / 기타: 협업 도구(JIRA))"
    "summary_keywords": "#열정적 #창의적 #꼼꼼함"
    """

    # Function to get string response with retries
    async def get_string_response(prompt, retries):
        for attempt in range(retries):
            try:
                # Interact with GPT using OpenAI's chat completions
                completion = openai.chat.completions.create(
                    model=os.getenv("gpt"),
                    messages=[
                        {"role": "system", "content": "Perform the task of extracting information from a resume received in PDF format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

                # Extract the response content
                response_content = completion.choices[0].message.content.strip()

                # Return the response as a string (not JSON)
                return response_content
            
            except KeyError:
                if attempt < retries - 1:
                    print(f"Attempt {attempt + 1} failed. Retrying...")
                    await asyncio.sleep(2 ** attempt)  # 비동기 대기
                else:
                    # Raise an error after the final attempt
                    raise ValueError("Failed to get response after multiple attempts.")
    
    # Get the string response with retries
    summation = await get_string_response(prompt, max_retries)

    # Return the result
    return summation


