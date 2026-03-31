# """
# LLM wrapper using an API-based model (OpenAI-style).

# This module exposes a single function:
#     generate(prompt: str) -> str

# So the rest of the RAG pipeline does NOT need to change.
# """

# import os
# from openai import OpenAI


# # 1. Load API key securely
# # Set this once in your environment:
# # export OPENAI_API_KEY="your-key"
# # or on Windows (PowerShell):
# # setx OPENAI_API_KEY "your-key"

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )


# # 2. Generate function
# def generate(prompt: str) -> str:
#     """
#     Sends the prompt to the API LLM and returns the response text.

#     Parameters:
#         prompt (str): Fully constructed RAG prompt

#     Returns:
#         str: Model-generated answer
#     """

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",  # or gpt-4, gpt-3.5-turbo, etc.
#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a Payment Failure Analysis Assistant. "
#                     "Follow instructions strictly and do not hallucinate."
#                 ),
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         temperature=0.2,          # deterministic output
#         max_tokens=400            # control output length
#     )

#     # Extract only the assistant's message
#     return response.choices[0].message.content.strip()
