import ast
import torch
import requests
from io import BytesIO
from PIL import Image, ImageDraw

# Pastikan Anda punya qwen_vl_utils.py atau modul setara yang menyediakan process_vision_info
# pip install transformers Pillow
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def draw_point(image_input, point=None, radius=5):
    """
    Menampilkan gambar sambil menandai titik (point) dengan lingkaran merah.
    `point` diasumsikan [x_norm, y_norm] dalam skala 0..1 relatif ke gambar.
    """
    # Load image
    if isinstance(image_input, str):
        if image_input.startswith('http'):
            image = Image.open(BytesIO(requests.get(image_input).content))
        else:
            image = Image.open(image_input)
    else:
        image = image_input

    # Gambar lingkaran merah pada koordinat yang dihitung
    if point:
        x, y = point[0] * image.width, point[1] * image.height
        draw = ImageDraw.Draw(image)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')

    # Tampilkan gambar dengan PIL
    image.show()

# ------------------------------------------------------------------------
# Bagian utama kode
# ------------------------------------------------------------------------

# Muat model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "showlab/ShowUI-2B",
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Pastikan Anda punya GPU atau bisa diganti "cpu"
)

# Tentukan batas min/max pixels
min_pixels = 256 * 28 * 28
max_pixels = 1344 * 28 * 28

# Muat processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", 
                                          min_pixels=min_pixels, 
                                          max_pixels=max_pixels)

# Ganti 'examples/gambar1.png' dengan path/gambar lain sesuai kebutuhan
img_url = 'examples/gambar2.png'
# query = "Dover"
query = input("Input yang ingin dicari: ")

_SYSTEM = (
    "Based on the screenshot of the page, I give a text description and you give "
    "its corresponding location. The coordinate represents a clickable location "
    "[x, y] for an element, which is a relative coordinate on the screenshot, "
    "scaled from 0 to 1."
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": _SYSTEM},
            {
                "type": "image",
                "image": img_url,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
            {"type": "text", "text": query},
        ],
    }
]

# Siapkan teks prompt
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)

# Proses image/video input
image_inputs, video_inputs = process_vision_info(messages)

# Buat input tensor untuk model
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Pastikan device sama dengan device_map, misalnya GPU
inputs = inputs.to("cuda")

# Lakukan inferensi
generated_ids = model.generate(**inputs, max_new_tokens=128)

# Potong token hasil generasi agar tidak menyertakan token input
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# Decode output menjadi string
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

# Output dari model diharapkan berupa string yang merepresentasikan list Python
# Contoh: "[0.73, 0.21]"
click_xy = ast.literal_eval(output_text)

# Gambar titik merah pada screenshot
draw_point(img_url, click_xy, radius=10)
