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
    max_tokens=512
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

Output Format:
This is just a out oromat , no of explatation can be anynumber based on the text
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
{format_instruction}
"""


document_text='''my day is not bad however i enjoy a lot in office 
but professionaly my day sucks. let see what will happen tomorrow and by the way tomorrow is my birthday, i am sad becasuse i need to go to office.
In office i become happy when i play table tennis'''

from langchain_core.output_parsers import JsonOutputParser

json_parser=JsonOutputParser()

prompt_template=PromptTemplate(
    template=prompt,
    input_variables=['document_text'],
    partial_variables={'format_instruction': json_parser.get_format_instructions()}
)

pt_result=prompt_template.invoke({'document_text': document_text})


model_output=model.invoke(pt_result)
final_result=json_parser.parse(model_output.content)
print(final_result)


