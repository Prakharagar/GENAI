from langchain.prompts import PromptTemplate, PipelinePromptTemplate

# Define individual prompt templates
intro_template = PromptTemplate.from_template("Introduce the topic of {topic}.")
detail_template = PromptTemplate.from_template("Provide details about {introduction}.")
conclusion_template = PromptTemplate.from_template("Summarize the key points of {details}.")

# Create a PipelinePromptTemplate
pipeline_prompt = PipelinePromptTemplate(
    pipeline_prompts=[
        ("introduction", intro_template),
        ("details", detail_template),
    ],
    final_prompt=conclusion_template,
)

# Format the pipeline prompt
formatted_prompt = pipeline_prompt.format(topic="Large Language Models")
print(formatted_prompt)