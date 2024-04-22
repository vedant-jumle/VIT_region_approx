import os 
os.environ["KERAS_BACKEND"] = "torch"

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import einops
from keras import ops
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    model.to(device)

    return model, processor, tokenizer

    
def preprocess_vit(image, processor):
    if type(image) == str:
        image = Image.open(image)
        image = image.convert(mode="RGB")
    
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    return pixel_values

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def extract_roi(image, processor, tokenizer, model):
    
    def model_function(input_ids, pixel_values):
        with torch.no_grad():
            outputs = model(pixel_values, torch.tensor([input_ids]).cuda(), output_attentions=True)
            cross_attention = outputs.cross_attentions
            encoder_attention = outputs.encoder_attentions
            decoder_attention = outputs.decoder_attentions
            logits = torch.squeeze(outputs.logits[-1, -1].cpu()).numpy()

        return logits, cross_attention, encoder_attention, decoder_attention
    
    
    pixel_values = preprocess_vit(image, processor)

    input_start_token = tokenizer("<|endoftext|>", return_tensors='pt')
    input_ids = input_start_token['input_ids'].to('cuda')
    attention_mask = input_start_token['attention_mask'].to('cuda')

    sampled_ids = input_ids.cpu().numpy().tolist()[0]
    start_length = len(sampled_ids)
    max_length = 20
    attention_weights = {
        "cross_attention": [],
        "encoder_attention": [],
        "decoder_attention": []
    }

    while len(sampled_ids) <= max_length+start_length:
        logits, cross_attention, encoder_attention, decoder_attention = model_function(sampled_ids, pixel_values)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        sampled_idx = np.argmax(probs)

        attention_weights['cross_attention'].append(np.array([item.cpu().numpy()[0] for item in cross_attention]))
        attention_weights['encoder_attention'].append(np.array([item.cpu().numpy()[0] for item in encoder_attention]))
        attention_weights['decoder_attention'].append(np.array([item.cpu().numpy()[0] for item in decoder_attention]))

        sampled_ids.append(sampled_idx)

        print(tokenizer.decode(sampled_ids), end='\r')

        if sampled_idx == 50256:
            break
    
    
    patch_size = 16
    w_featmap = 224 // patch_size
    h_featmap = 224 // patch_size
    num_heads = 12

    in1k_mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255]) # image darking filters
    in1k_std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])

    image_np = np.transpose(pixel_values.cpu().numpy()[0], (1, 2, 0))

    preprocessed_img_orig = (image_np * in1k_std) + in1k_mean   
    preprocessed_img_orig = preprocessed_img_orig / 255.0
    preprocessed_img_orig = ops.convert_to_numpy(ops.clip(preprocessed_img_orig, 0.0, 1.0))

    n_cols = 4
    n_rows = (len(sampled_ids)//4)+1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(13, 13))
    token_count = 0

    for i in range(n_rows):
        for j in range(n_cols):
            if token_count < len(sampled_ids)- 1:
                token = sampled_ids[token_count]
                token_value = tokenizer.decode(token)

                attention_scores = attention_weights['cross_attention'][token_count][-1, :, -1, 1:] # [last layer, all heads, first context token, ignore cls]
                attention_scores = np.reshape(attention_scores, (num_heads, w_featmap, h_featmap))
                attention_scores = np.transpose(attention_scores, (1, 2, 0))
                attention_scores = ops.image.resize(attention_scores, size=(h_featmap * patch_size, w_featmap * patch_size)).cpu().numpy()
                average_attention_over_heads = np.mean(attention_scores, axis=-1)

                axes[i, j].imshow(preprocessed_img_orig)
                axes[i, j].imshow(average_attention_over_heads, cmap="inferno", alpha=0.6)
                axes[i, j].title.set_text(f"token: {token_value}")
                axes[i, j].axis("off")
                token_count += 1

    caption = tokenizer.decode(sampled_ids[1:-1])

    return fig2img(fig), caption

def square_crop(image, n):
    """
    Squares an image by cropping n pixels from each side.

    Args:
        image: The PIL.Image object to be cropped.
        n: The number of pixels to crop from each side (left, right, top, bottom).

    Returns:
        A new PIL.Image object with a square crop.
    """
    width, height = image.size
    # Get the minimum dimension to ensure a square crop
    crop_size = min(width - 2 * n, height - 2 * n)

    # Handle cases where the image is already square or smaller than desired crop size
    if crop_size <= 0:
        return image.copy()  # Return a copy of the original image

    # Calculate the top-left corner of the square crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2

    # Create the crop box (left, top, right, bottom)
    box = (left, top, left + crop_size, top + crop_size)

    # Crop the image and return the new square image
    return image.crop(box)

    # Example usage
    image = Image.open("path/to/your/image.jpg")
    cropped_image = square_crop(image, 10)  # Crop 10 pixels from each side
    cropped_image.show()  # Display the cropped image