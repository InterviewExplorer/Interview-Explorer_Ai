import openai
from dotenv import load_dotenv
import pdfplumber
import os
import json
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
    You are tasked with organizing the resume from the PDF file.

    Instructions:
    1. Extract all key-value pairs from the PDF's structured information.
    2. Output the values in a simplified format, where each key-value pair is on a new line.
    3. Example format:
    "name": "구예성",
    "date_of_birth": "2001-07-28",
    4. Do not include any nested JSON structures. Flatten all information into a list of key-value pairs.
    5. Exclude keys such as "personal_info" or "skills" headers, only show individual key-value pairs.
    6. Follow this format for all data, including personal info, projects, education, etc.
    7. Ensure the output is properly formatted, with each key and value appearing on its own line.
    8. Analyze the person's personality, tendencies, and characteristics in the PDF file, define words that can express these, and create within 5 words. Keywords must be words that can identify the personality, personality, and characteristics in Korean.

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