import openai
from dotenv import load_dotenv
import pdfplumber
import os
import asyncio

async def pdf(pdf_path, max_retries=3):
    # Load environment variables
    load_dotenv()

    # Set up OpenAI API key
    api_key = os.getenv("API_KEY2")
    if api_key is None:
        raise ValueError("API_KEY is missing.")
    
    openai.api_key = api_key

    # Extract text from PDF
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    # Prepare prompt for GPT, including the extracted PDF content
    prompt = f"""
    You are tasked with summarizing and organizing a resume received in PDF format.

    Instructions:

    1. Extract key information from the PDF file.
    2. Organize the information in a simplified key-value pair format, with each pair on a new line.
    3. Include only the following key information:
        - Name
        - Date of birth
        - Technical skills (divided into back-end and front-end)
        - Work experience
        - Number of projects
        - description of each project
    4. Format the output as follows: "key": "value"
    5. Analyze the person's personality, tendencies, and characteristics based on the PDF file.
    6. Define up to 5 keywords that express these traits. Each keyword should be a single word in Korean that captures an aspect of the person's personality, character, or distinctive features.
    7. Output all information in Korean only.
    8. Express that there is no content that is not there. Do not generate it randomly and make guesses.

    Example format:
    "name": "구예성"
    "date_of_birth": "2001-07-28"
    "technical_skills": "backend:  / frontend:  / ai:  / tools:  / db:"
    "work_experience": "2 years"
    "number_of_projects": "3"
    "project_description": "쇼핑몰, 교육플랫폼, 블로그"
    "summary_keywords": "열정적, 창의적, 꼼꼼함"

    Precautions:
        - For the project_description, briefly analyze each project within 10 characters and define the nature of the site.
        - For example, you can express it as "Online shop", "Edu platform", "Personal blog", etc.
        - If there are multiple projects, list them separated by commas.
        - If there is no content for technical_skills, don't even create a category.
        - Do not add “,” at the end of the value.

PDF content:
{text}

Ensure all text is properly formatted and in Korean.

    PDF content:
    {text}

    Remember to maintain a structure that resembles JSON and ensure all text is properly escaped for formatting.
    """

    # Function to get string response with retries
    async def get_string_response(prompt, retries):
        for attempt in range(retries):
            try:
                # Interact with GPT using OpenAI's chat completions
                completion = openai.chat.completions.create(
                    model=os.getenv("gpt"),
                    messages=[
                        {"role": "system", "content": "You are a professional interviewer specializing in assessing resumes."},
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