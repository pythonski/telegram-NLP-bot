import os
import telebot
from google.cloud import speech
import requests
import subprocess
from datetime import datetime
from retriever import add_entry_to_diary
from pipelines import qa_pipeline, classify_type, helper
import re

# Credentials
TOKEN = token # insert your google speech here
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google.json"

bot = telebot.TeleBot(TOKEN)

def transcribe_audio_google(file_path):
    client = speech.SpeechClient()

    with open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,  # needs the correct encoding
        language_code="en",
    )

    response = client.recognize(config=config, audio=audio)
    return response


user_states = {}

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_message = (
        "Welcome to the Project Diary Bot!\n\n"
        "You can input diary entries (by text or audio message) or ask questions about the project.\n\n"
        "Use /transcript to visualize a specific past entry, /entry to add a new diary entry, /question to ask a question, or /summary to generate summaries (warning: this is an experimental function and will return a .md file).\n\n"
        "Note: audio messages are always considered potential entries. If you have a question, you should type it out for now!\n\n"
        "How can I assist you?"
    )
    bot.reply_to(message, welcome_message)


@bot.message_handler(commands=['summary'])
def handle_summary(message):
    bot.send_message(message.chat.id, 'Generating summaries...')
    diary_entries = read_all_entries()
    summaries = []
    
    for entry in diary_entries:
        summary = summarize(entry=entry["text"]).summary
        summaries.append(f"### {entry['date']}\n\n{summary}\n\n")
    
    summary_content = "\n".join(summaries)
    summary_file_path = "diary_summary.md"
    
    with open(summary_file_path, "w") as file:
        file.write(summary_content)
    
    with open(summary_file_path, "rb") as file:
        bot.send_document(message.chat.id, file)


@bot.message_handler(commands=['entry'])
def handle_entry_command(message):
    chat_id = message.chat.id
    user_states[chat_id] = {'state': 'waiting_for_entry'}
    bot.send_message(chat_id, "Please provide the diary entry text.")


@bot.message_handler(commands=['transcript'])
def handle_transcript_command(message):
    chat_id = message.chat.id
    diary_entries = read_all_entries()
    if not diary_entries:
        bot.send_message(chat_id, "No entries found in the diary.")
        return

    dates = [entry["date"] for entry in diary_entries]
    dates_message = "Available dates:\n" + "\n".join(dates)
    bot.send_message(chat_id, dates_message)
    
    user_states[chat_id] = {'state': 'waiting_for_transcript_date', 'dates': dates}

    
@bot.message_handler(commands=['question'])
def handle_question_command(message):
    chat_id = message.chat.id
    user_states[chat_id] = {'state': 'waiting_for_question'}
    bot.send_message(chat_id, "Please provide the question.")


@bot.message_handler(content_types=['audio', 'voice'])
def handle_audio_message(message):
    chat_id = message.chat.id
    try:
        bot.reply_to(message, "Processing your voice message...")
        if message.voice:
            file_info = bot.get_file(message.voice.file_id)
        else:
            file_info = bot.get_file(message.audio.file_id)

        file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}"
        ogg_file_path = f"{file_info.file_id}.ogg"
        
        response = requests.get(file_url)
        with open(ogg_file_path, 'wb') as f:
            f.write(response.content)

        # Convert OGG to FLAC
        flac_file_path = ogg_file_path.replace(".ogg", ".flac")
        subprocess.run(['ffmpeg', '-i', ogg_file_path, flac_file_path])

        transcription = transcribe_audio_google(flac_file_path).results[0].alternatives[0].transcript
        bot.reply_to(message, f"This is the transcription of your message: {transcription}")
        
        # Save the transcription and update the user state
        user_states[chat_id] = {'state': 'waiting_for_confirmation', 'transcription': transcription}
        
        # Ask permission
        bot.send_message(chat_id, f"Do you want to add this transcription to your journal (date: {datetime.now().strftime('%d-%m-%y')})? Reply with 'yes' or 'no'.")

    except Exception as e:
        bot.reply_to(message, f"An error occurred: {e}")

    finally:
        # Clean up the temporary files
        if os.path.exists(ogg_file_path):
            os.remove(ogg_file_path)
        if os.path.exists(flac_file_path):
            os.remove(flac_file_path)

    
@bot.message_handler(func=lambda message: True)
def handle_text_messages(message):
    chat_id = message.chat.id
    
    if chat_id in user_states:
        state = user_states[chat_id]['state']
        
        if state == 'waiting_for_entry':
            add_entry_to_diary(message.text)
            bot.send_message(chat_id, "Diary entry added successfully.")
            user_states.pop(chat_id, None)
        
        elif state == 'waiting_for_question':
            bot.send_message(chat_id, "Generating answer...")
            prediction = qa_pipeline(message.text)
            response = helper(question=message.text, original_answer=prediction.answer)
            bot.reply_to(message, f"Answer: {response.helpful_answer}")
            user_states.pop(chat_id, None)
        
        elif state == 'waiting_for_confirmation':
            if message.text.lower() == 'yes':
                transcription = user_states.pop(chat_id, None)['transcription']
                if transcription:
                    add_entry_to_diary(transcription)
                    bot.send_message(chat_id, "Diary entry added successfully.")
            
            elif message.text.lower() == 'no':
                bot.send_message(chat_id, "The transcription was not added to the diary.")
                user_states.pop(chat_id, None)
            else:
                bot.send_message(chat_id, "Please reply with 'yes' or 'no'.")
        
        elif state == 'waiting_for_transcript_date':
            requested_date = message.text
            if requested_date in user_states[chat_id]['dates']:
                entry = get_entry_by_date(requested_date)
                bot.send_message(chat_id, f"Entry for {requested_date}:\n\n{entry}")
            else:
                bot.send_message(chat_id, "Invalid date. Please choose a valid date from the list.")
            user_states.pop(chat_id, None)
    else:
        bot.send_message(chat_id, "Please use /entry to add a new diary entry, /question to ask a question, or /transcript to read an entry.")


def get_entry_by_date(requested_date, file_path='project_diary.txt'):
    diary_entries = read_all_entries(file_path)
    
    for entry in diary_entries:
        if entry["date"] == requested_date:
            return entry["text"]
    
    return "No entry found for the specified date."
    

def read_all_entries(file_path='project_diary.txt'):
    with open(file_path, 'r') as file:
        content = file.read()
    
    entries = content.split('Date:')
    entries = [entry.strip() for entry in entries if entry.strip()]
    diary_entries = []
    
    for entry in entries:
        date_match = re.match(r'(\d{2}-\d{2}-\d{4})', entry)
        if date_match:
            date = date_match.group(1)
            text = entry.replace(date, '').strip()
            diary_entries.append({"date": date, "text": text})
    
    return diary_entries
