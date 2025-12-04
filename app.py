import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration


# load processor and model
PROCESSOR = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def caption_image(input_img: np.ndarray) -> str:
    raw_image = Image.fromarray(input_img).convert("RGB")

    inputs = PROCESSOR(raw_image, return_tensors="pt")

    outputs = MODEL.generate(**inputs, max_length=50)

    caption = PROCESSOR.decode(outputs[0], skip_special_tokens=True)

    return caption


interface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning",
    description="A simple web app for generating captions for images."
)

interface.launch()
