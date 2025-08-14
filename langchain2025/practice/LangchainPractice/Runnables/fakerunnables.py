from config_loader import *
from abc import ABC,abstractmethod

class runnables:
    @abstractmethod
    def invoke(data):
        pass

class llm_model(runnables):
    def __init__(self,model_name:str,config ={'max_tokens':512,'temperature':1}):
        if model_name == 'ChatOpenAI':
            from langchain_openai import ChatOpenAI      
            self.model =ChatOpenAI(**config)
        else:
            raise ValueError("Invalid model name")
        
    def invoke(self,data):
        response=self.model.invoke(data)
        return response
    
class prmttemplate(runnables):
    def __init__(self,prompt,input_variables):
        self.prompt=prompt
        self.input_variables=input_variables

    def invoke(self,data):
        return self.prompt.format(**data)

obj=llm_model('ChatOpenAI',{'max_tokens':512,'temperature':1})
print(obj.invoke('hi'))
