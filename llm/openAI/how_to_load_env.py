from dotenv import load_dotenv
import os
load_dotenv()

token=os.getenv('token')
print(token)