import openai
from dotenv import load_dotenv
import pdfplumber
import os
import asyncio
from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta
import re
import calendar

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
    - Content in sections labeled as 'Experience' or 'Work Experience' is not considered a project.
    - The number of projects must be expressed only in numerical form.

    # Task4 : Project description
    - Content in sections labeled as 'Experience' or 'Work Experience' is not considered a project.
    - From the resume content, summarize the purpose of each project in a few words, based on the project name or description provided in the project section.

    # Task5 : Work experience    
    - Extract all periods corresponding to experience and list them as a string, separated by "/".
    - Include as work experience if it contains both the term '경력' and the company name.
    - If there is no information about work experience, indicate it as '없음'.
    - Internships or competition activities are excluded when calculating work experience.

    Example:
    2017.03- 현재 재직 중
    2012.08 ~ 2017.02
    2007.03-2012.07
    Total work experience = 2017.03 ~ 2024.10 / 2012.08 ~ 2017.02 / 2007.03 ~ 2012.07

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
    "work_experience": "2017.03 ~ 2024.10 / 2012.08 ~ 2017.02 / 2007.03 ~ 2012.07"
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

                response_content = completion.choices[0].message.content.strip()

                # work_experience 추출
                work_experience = extract_work_experience(response_content)
                
                if work_experience:
                    # 계산된 work_experience로 대체
                    calculated_experience = calculate_work_experience(work_experience)
                    response_content = re.sub(
                        r'"work_experience":\s*"[^"]*"',
                        f'"work_experience": "{calculated_experience}"',
                        response_content
                    )

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

def extract_work_experience(response_content):
    match = re.search(r'"work_experience":\s*"([^"]*)"', response_content)
    if match:
        return match.group(1)
    return None

def calculate_work_experience(work_experience):
    print("@@@@work_experience", work_experience)

    experiences = work_experience.split('/')
    current_date = datetime.now().strftime("%Y.%m")
    total_years = 0
    total_months = 0

    for i, experience in enumerate(experiences, 1):
        if any(keyword in experience for keyword in ['현재', '현재 재직 중', '현재 재직중']):
            experience = experience.replace('현재', current_date).replace('재직 중', '').replace('재직중', '').strip()
        
        dates = experience.split('~')
        if len(dates) == 2:
            start_date = dates[0].strip()
            end_date = dates[1].strip()

            # 시작 날짜와 종료 날짜를 년과 월로 나누기
            start_year, start_month = map(int, start_date.split('.'))
            end_year, end_month = map(int, end_date.split('.'))

            # 경력 기간 계산
            years = end_year - start_year
            months = end_month - start_month

            if months < 0:
                years -= 1
                months += 12

            total_years += years
            total_months += months
            print(f"경력 {i}: {years}년 {months}개월")

        else:
            print(f"경력 {i}: {experience} (올바른 형식이 아님)")

    # 총 개월 수 계산 및 년/월로 변환
    additional_years, remaining_months = divmod(total_months, 12)
    final_years = total_years + additional_years
    final_months = remaining_months

    return f"{final_years}년 {final_months}개월"

