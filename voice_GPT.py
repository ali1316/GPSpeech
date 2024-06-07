import speech_recognition as sr
import os
import subprocess
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
_ = load_dotenv(find_dotenv())

def ConnectToAzure():
    """
    desc:
        Function connects to langchain AzureOpenAI
    return: 
        model of llm
    """
    ## Keys ##
    OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

    model = AzureChatOpenAI(
        openai_api_base=OPENAI_API_BASE,
        openai_api_version=OPENAI_API_VERSION,
        azure_deployment=DEPLOYMENT_NAME,
        openai_api_key=OPENAI_API_KEY,
        openai_api_type=OPENAI_API_TYPE,
    )
    return model

def ConversationInput():
    _DEFAULT_TEMPLATE = """
    You are a helpful speech assistant that answers all the human's questions.
    
    Current conversation:
    New human question: {input}
    Response:"""

    prompt = PromptTemplate(
        input_variables=["input"], template=_DEFAULT_TEMPLATE
    )

    conversation = LLMChain(
    llm=ConnectToAzure(),
    prompt=prompt,
    verbose=False,
    )
    return conversation

WAKE_WORD = "hey bro"

recognizer = sr.Recognizer()
microphone = sr.Microphone()

def listen_for_wakeword():
  """Listens for the wake word only"""
  with microphone as source:
    print("Say 'hey bro' so I can assist you...")
    audio = recognizer.listen(source, phrase_time_limit=5)  # Limit listening time  
  
  try:
    text = recognizer.recognize_google(audio).lower()  # Convert to lowercase
    if text.startswith(WAKE_WORD):
      return text
    else:
      return None
  except sr.UnknownValueError:
    return None  # Ignore unrecognizable audio

def listen_after_wakeword():
  """Listens and recognizes speech after wake word"""
  with microphone as source:
    print("Talk bro what do you want...")
    audio = recognizer.listen(source)
  
  try:
    text = recognizer.recognize_google(audio)
    return text.lower()
  
  except sr.UnknownValueError:
    print("Sorry, could not understand audio")
    return None

def run_app():
  text = listen_for_wakeword()
  if text:
    print("Hey! I'm awake.")
    after_wakeword = listen_after_wakeword()
    print("You said: " + after_wakeword)
    
    #commands:
    if after_wakeword == "open calculator":
      print("opening calculator")
      subprocess.call('calc.exe')
    elif after_wakeword == "open notepad":
      print("opening notepad")
      subprocess.call('notepad.exe')
    elif after_wakeword == "open cmd":
      print("opening CMD")
      subprocess.call('cmd.exe')
    elif after_wakeword == "open anki":
      print("opening Anki")
      subprocess.call('D://Anki//anki.exe')
    elif after_wakeword == "open zoom":
      print("opening Zoom")
      subprocess.call('C://Users//Ahmed//AppData//Roaming//Zoom//bin//Zoom.exe')

    #gpt QnA
    elif after_wakeword:
      print("Processing...")
      
      conversation = ConversationInput()
      response = conversation.predict(input = after_wakeword)
      
      print("Response: ", response)

run_app()