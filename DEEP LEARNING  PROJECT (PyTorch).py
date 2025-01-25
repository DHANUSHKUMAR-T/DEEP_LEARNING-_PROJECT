import torch
import pywhatkit
import datetime
import wikipedia
import pyjokes
import os
from cv2 import VideoCapture, destroyWindow, imshow, imwrite
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import warnings

warnings.filterwarnings("ignore", message="Unused or unrecognized kwargs: padding")

model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

gen_kwargs = {'max_length': 16, 'num_beams': 4}

def help():
    print("""
Available Commands:
- play <song_name>: Play a song on YouTube.
- time: Show the current time.
- today date
- yesterday date 
- tomorrow date
- search <topic>: Get a summary from Wikipedia.
- joke: Hear a funny joke.
- capture: Capture an image and generate a caption.
- analyze <image_file>: Analyze a saved image and generate a caption.
- list files: List all files in the current directory.
- open <file_name>: Open a file from the current directory.
- calculator: Launch a simple calculator.
- exit: Quit the assistant.
    """)

def predict_caption(image_path):
    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")

        pixel_values = feature_extractor(images=img, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(device)
        attention_mask = torch.ones(pixel_values.shape[:2], dtype=torch.long).to(device)
        output_ids = model.generate(
            pixel_values, 
            attention_mask=attention_mask, 
            **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        caption = preds[0].strip()
        print(f"Caption for the image: {caption}")
        return caption
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def capture_image_and_caption():
    print("Capturing image...")
    cam = VideoCapture(0)
    ret, frame = cam.read()
    if ret:
        image_path = "captured_image.png"
        imwrite(image_path, frame)
        print(f"Image saved as '{image_path}'.")
        predict_caption(image_path)
    else:
        print("Error: Unable to capture image.")
    cam.release()
    destroyWindow("Captured Image")

def list():
    print("Files in Directory:")
    for file in os.listdir():
        print("-", file)

def open_file(file_name):
    try:
        os.startfile(file_name)
        print(f"Opened file: {file_name}")
    except Exception as e:
        print(f"Error opening file '{file_name}': {e}")

def calculator():
    print("Calculator Mode (Type 'back' to exit):")
    while True:
        expr = input("Enter operation: ").strip()
        if expr.lower() == 'back':
            break
        try:
            print(f"Result: {eval(expr)}")
        except Exception as e:
            print("Invalid Input:", e)

def run():
    print("Welcome! I am Assista. Type 'help' for commands.")
    while True:
        command = input("\nYour command: ").lower().strip()
        
        if command.startswith("play"):
            song = command.replace("play", "").strip()
            pywhatkit.playonyt(song)
            print(f"playing {song}")

        elif command == "time":
            print("Current time:", datetime.datetime.now().strftime("%I:%M %p"))

        elif "date" in command:
            today = datetime.date.today()
            if "yesterday" in command:
                print("Yesterday's date:", today - datetime.timedelta(days=1))
            elif "tomorrow" in command:
                print("Tomorrow's date:", today + datetime.timedelta(days=1))
            else:
                print("Today's date:", today)

        elif command.startswith("search"):
            topic = command.replace("search", "").strip()
            try:
                summary = wikipedia.summary(topic, sentences=1)
                print("Wikipedia:", summary)
            except Exception:
                print("No summary found for the topic.")

        elif command == "joke":
            print("Joke:", pyjokes.get_joke())

        elif command == "capture":
            capture_image_and_caption()

        elif command.startswith("analyze"):
            image_file = command.replace("analyze", "").strip()
            if os.path.exists(image_file):
                predict_caption(image_file)
            else:
                print(f"File '{image_file}' not found.")

        elif command == "list files":
            list()

        elif command.startswith("open"):
            file_name = command.replace("open", "").strip()
            open_file(file_name)

        elif command == "calculator":
            calculator()

        elif command == "help":
            help()

        elif command == "exit":
            print("Goodbye!")
            break

        else:
            print("Unknown command. Type 'help' for options.")

run()