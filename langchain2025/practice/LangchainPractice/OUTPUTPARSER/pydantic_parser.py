from config_loader import *
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

'''
1. create model
2. design prompt
    2.a. give/extgract input which is required for design template
3. invoke prompt tempate
4. structure output to the model
4. invoke model
'''

#create model

model=ChatOpenAI(
    temperature=0,
    max_tokens=512,
    seed=1
)

prompt = """
You are a strict and deterministic sentiment analysis parser.

Based ONLY on the following input text (provided in the 'Text:' section), you must extract any number of sentiment explanations and corresponding sentiments.

You are not allowed to infer, analyze, or reinterpret the meaning of the text. Instead, parse it directly and follow the exact structure and calculation method described below.

INSTRUCTIONS:
1. Extract explanations step by step for each phrase from the text and assign each a sentiment: Positive, Neutral, or Negative.
2. For each explanation, assign probability values for Positive, Neutral, and Negative — each set must add up to exactly 100%.
3. Count how many of the explanations are Positive, Neutral, and Negative.
4. Calculate the **Overall Probability** by averaging the individual explanation probabilities across each sentiment.
   - Example: If Explanation 1 has Positive=40%, 2 has Positive=60%, and 3 has Positive=50%, then overall Positive = (40+60+50)/3 = 50%.
5. Determine the Overall Sentiment as the sentiment with the highest overall probability.
6. Provide a short justification in the Overall Explanation based on step-by-step calculations.
7. DO NOT add any extra interpretation, summary, or inferred reasoning beyond what is described.

Input Text:
\"\"\"{document_text}\"\"\"
{format_instruction}
"""


document_text='''my day is not bad however i enjoy a lot in office 
but professionaly my day sucks. let see what will happen tomorrow and by the way tomorrow is my birthday, i am sad becasuse i need to go to office.
In office i become happy when i play table tennis'''

from langchain_core.output_parsers import PydanticOutputParser 
from typing import Literal, List
from pydantic import Field, BaseModel


class Probability(BaseModel):
    positive: float = Field(..., alias="Positive")
    neutral: float = Field(..., alias="Neutral")
    negative: float = Field(..., alias="Negative")

class Explanation(BaseModel):
    explanation: str
    sentiment: Literal["Positive", "Neutral", "Negative"]
    probability: Probability

class AverageProbability(BaseModel):
    positive: float = Field(..., alias="Positive")
    neutral: float = Field(..., alias="Neutral")
    negative: float = Field(..., alias="Negative")

class OverallExplanation(BaseModel):
    number_of_positive: int = Field(..., alias="Number of Positive")
    number_of_neutral: int = Field(..., alias="Number of Neutral")
    number_of_negative: int = Field(..., alias="Number of Negative")
    average_probabilities: AverageProbability = Field(..., alias="Average Probabilities")
    final_decision: str = Field(..., alias="Final Decision")

class SentimentReport(BaseModel):
    explanations: List[Explanation]
    overall_explanation: OverallExplanation = Field(..., alias="Overall Explanation")
    overall_sentiment: Literal["Positive", "Neutral", "Negative"] = Field(..., alias="Overall Sentiment")
    overall_probability: float = Field(..., alias="Overall Probability")

pydantic_parser=PydanticOutputParser(pydantic_object=SentimentReport)

prompt_template=PromptTemplate(
    template=prompt,
    input_variables=['document_text'],
    partial_variables={'format_instruction': pydantic_parser.get_format_instructions()}
)

#create chain

chain= prompt_template|model|pydantic_parser

result=chain.invoke({'document_text':document_text})
print(result)






