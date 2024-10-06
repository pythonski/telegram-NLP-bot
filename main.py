import dspy
from bot import bot
from lm import GroqLM
import subprocess

def is_process_running(process_name):
    try:
        output = subprocess.check_output(f"pgrep -f '{process_name}'", shell=True)
        return True
    
    except subprocess.CalledProcessError:
        return False


def start_process(command):
    subprocess.Popen(command, shell=True)


def main():

    # Uncomment to set up local model
    #process_name = "ollama run llama3" 
    #start_command = "ollama run llama3" 

    #if not is_process_running(process_name):
    #    print(f"{process_name} is not running. Starting it now...")
    #    start_process(start_command)
    #else:
    #    print(f"{process_name} is already running.")

    # Remote model as alternative
    language_model = GroqLM(api_key=API_KEY, max_tokens=1000) # put your API key here
    dspy.configure(lm=language_model)

    # Activate bot
    print("Starting Telegram bot...")
    bot.polling()

if __name__ == "__main__":
    main()
