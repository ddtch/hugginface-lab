# Use a pipeline as a high-level helper
from transformers import AutoProcessor, BlipForQuestionAnswering
import warnings
from transformers.utils import logging
from PIL import Image

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

model_name="Salesforce/blip-vqa-base"
# For image-to-text generation
processor = AutoProcessor.from_pretrained(model_name)
model = BlipForQuestionAnswering.from_pretrained(model_name)

raw_image = Image.open("./assets/images/sample.jpeg", 'r').convert('RGB')

question = input("Enter a question: ")

inputs = processor(images=raw_image, text=question, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=100)
generated_answer = processor.decode(out[0], skip_special_tokens=True)

print(f"Generated answer: -{generated_answer}");

while True:
    try:
        question = input("Enter a question: ")
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    
    if question == "exit":
        break
    inputs = processor(images=raw_image, text=question, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=100)
    generated_answer = processor.decode(out[0], skip_special_tokens=True)
    print(f"Generated answer: -{generated_answer}");
    print("--------------------------------")
    print("--------------------------------")
