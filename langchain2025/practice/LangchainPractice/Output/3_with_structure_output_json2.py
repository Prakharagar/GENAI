from config_loader import *
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel


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
    max_tokens=512
)

prompt = """
You are a strict and deterministic sentiment analysis parser.

Based ONLY on the following input text (provided in the 'Text:' section), you must extract exactly three sentiment explanations and corresponding sentiments.

You are not allowed to infer, analyze, or reinterpret the meaning of the text. Instead, parse it directly and follow the exact structure and calculation method described below.

INSTRUCTIONS:
1. Extract exactly 3 explanations from the text and assign each a sentiment: Positive, Neutral, or Negative.
2. For each explanation, assign probability values for Positive, Neutral, and Negative — each set must add up to exactly 100%.
3. Count how many of the 3 explanations are Positive, Neutral, and Negative.
4. Calculate the **Overall Probability** by averaging the individual explanation probabilities across each sentiment.
   - Example: If Explanation 1 has Positive=40%, 2 has Positive=60%, and 3 has Positive=50%, then overall Positive = (40+60+50)/3 = 50%.
5. Determine the Overall Sentiment as the sentiment with the highest overall probability.
6. Provide a short justification in the Overall Explanation based on step-by-step calculations.
7. DO NOT add any extra interpretation, summary, or inferred reasoning beyond what is described.

Input Text:
\"\"\"{document_text}\"\"\"

Output Format:

- Explanation 1: <sentence extracted or paraphrased from the text>
  Sentiment: <Positive/Neutral/Negative>
  Probability:
    - Positive: <xx.x>%
    - Neutral: <xx.x>%
    - Negative: <xx.x>%

- Explanation 2: ...
  Sentiment: ...
  Probability:
    - Positive: ...
    - Neutral: ...
    - Negative: ...

- Explanation 3: ...
  Sentiment: ...
  Probability:
    - Positive: ...
    - Neutral: ...
    - Negative: ...

Overall Explanation:
- Number of Positive: X
- Number of Neutral: Y
- Number of Negative: Z
- Average Probabilities:
    - Positive: <computed avg>%
    - Neutral: <computed avg>%
    - Negative: <computed avg>%
- Final Decision: Highest average probability determines Overall Sentiment.

Overall Sentiment: <Positive/Neutral/Negative>
Overall Probability: <xx.x>% (value of the highest average above for eh postive is 30,negative is 40 and neutral is 30 so the probality should be 40)
"""


document_text='''my day is not bad however i enjoy a lot in office 
but professionaly my day sucks'''

prompt_template=PromptTemplate(
    template=prompt,
    input_variables=['document_text']
)

pt_result=prompt_template.invoke({'document_text': document_text})

from pydantic import BaseModel, Field
from typing import Literal, List


from pydantic import BaseModel
from typing import Literal, List
from pprint import pprint


json_schema = {
  "title": "SentimentReport",
  "description": "Structured output for sentiment analysis report.",  # ✅ required
  "type": "object",
  "properties": {
    "explanations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "explanation": {"type": "string"},
          "sentiment": {
            "type": "string",
            "enum": ["Positive", "Neutral", "Negative"]
          },
          "probability": {
            "type": "object",
            "properties": {
              "Positive": {"type": "number"},
              "Neutral": {"type": "number"},
              "Negative": {"type": "number"}
            },
            "required": ["Positive", "Neutral", "Negative"]
          }
        },
        "required": ["explanation", "sentiment", "probability"]
      }
    },
    "Overall_Explanation": {
      "type": "object",
      "properties": {
        "Number_of_Positive": {"type": "integer"},
        "Number_of_Neutral": {"type": "integer"},
        "Number_of_Negative": {"type": "integer"},
        "Average_Probabilities": {
          "type": "object",
          "properties": {
            "Positive": {"type": "number"},
            "Neutral": {"type": "number"},
            "Negative": {"type": "number"}
          },
          "required": ["Positive", "Neutral", "Negative"]
        },
        "Final_Decision": {"type": "string"}
      },
      "required": [
        "Number_of_Positive",
        "Number_of_Neutral",
        "Number_of_Negative",
        "Average_Probabilities",
        "Final_Decision"
      ]
    },
    "Overall_Sentiment": {
      "type": "string",
      "enum": ["Positive", "Neutral", "Negative"]
    },
    "Overall_Probability": {"type": "number"}
  },
  "required": [
    "explanations",
    "Overall_Explanation",
    "Overall_Sentiment",
    "Overall_Probability"
  ]
}


# Output JSON Schema
#pprint(SentimentReport.model_json_schema())

pydantic_model=model.with_structured_output(json_schema)


model_output=pydantic_model.invoke(pt_result)
print(model_output)



