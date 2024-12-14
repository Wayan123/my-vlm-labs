import ast
import time
import torch
import requests
from io import BytesIO
from PIL import Image, ImageDraw

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

    # Tampilkan gambar
    image.show()

# -----------------------------------------------------------------------------
# 1) Load model & processor (bisa dipakai ulang untuk berbagai task)
# -----------------------------------------------------------------------------

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "showlab/ShowUI-2B",
    torch_dtype=torch.bfloat16,
    device_map="auto"  # "auto" mengasumsikan ada GPU; jika tak ada, gunakan "cpu"
)

# Tentukan batas min/max pixels
min_pixels = 256 * 28 * 28
max_pixels = 1344 * 28 * 28

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    min_pixels=min_pixels, 
    max_pixels=max_pixels
)

# -----------------------------------------------------------------------------
# 2) Bagian ShowUI: Generate koordinat klik berdasarkan screenshot
# -----------------------------------------------------------------------------

img_url_1 = 'examples/gambar1.png'   # Gambar pertama
query_1 = "Hull"

_SYSTEM = (
    "Based on the screenshot of the page, I give a text description and you give "
    "its corresponding location. The coordinate represents a clickable location "
    "[x, y] for an element, which is a relative coordinate on the screenshot, "
    "scaled from 0 to 1."
)

messages_1 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": _SYSTEM},
            {"type": "image", "image": img_url_1, "min_pixels": min_pixels, "max_pixels": max_pixels},
            {"type": "text", "text": query_1}
        ],
    }
]

text_1 = processor.apply_chat_template(
    messages_1, tokenize=False, add_generation_prompt=True,
)
image_inputs_1, video_inputs_1 = process_vision_info(messages_1)
inputs_1 = processor(
    text=[text_1],
    images=image_inputs_1,
    videos=video_inputs_1,
    padding=True,
    return_tensors="pt",
)
inputs_1 = inputs_1.to(model.device)

start_time_1 = time.time()
generated_ids_1 = model.generate(**inputs_1, max_new_tokens=128)
end_time_1 = time.time()

# Hitung waktu inferensi ShowUI
time_inference_1 = end_time_1 - start_time_1
print(f"Inference Time for ShowUI: {time_inference_1:.2f} seconds.")

generated_ids_trimmed_1 = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_1.input_ids, generated_ids_1)
]
output_text_1 = processor.batch_decode(
    generated_ids_trimmed_1, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("=== Output Koordinat ShowUI ===")
print(output_text_1)
# Misal output: "[0.73, 0.21]"

# Konversi string menjadi list Python
click_xy_1 = ast.literal_eval(output_text_1)

# Gambar titik merah pada screenshot pertama
draw_point(img_url_1, click_xy_1, radius=10)

# -----------------------------------------------------------------------------
# 3) Bagian UI Navigation: Generate aksi UI (CLICK/INPUT/etc.) menggunakan prompt
# -----------------------------------------------------------------------------

_NAV_SYSTEM = """You are an assistant trained to navigate the {_APP} screen. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
{_ACTION_SPACE}
"""

_NAV_FORMAT = """
Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

action_map = {
    'web': """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required. 
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
""",
    'phone': """
1. `INPUT`: Type a string into an element, value is not applicable and the position [x,y] is required. 
2. `SWIPE`: Swipe the screen, value is not applicable and the position [[x1,y1], [x2,y2]] is the start and end position of the swipe operation.
3. `TAP`: Tap on an element, value is not applicable and the position [x,y] is required.
4. `ANSWER`: Answer the question, value is the status (e.g., 'task complete') and the position is not applicable.
5. `ENTER`: Enter operation, value and position are not applicable.
"""
}

img_url_2 = 'examples/chrome.png'  # Gambar kedua
split = 'web'
system_prompt = _NAV_SYSTEM.format(_APP=split, _ACTION_SPACE=action_map[split])

query_2 = "Search the weather for the New York city."

messages_2 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": f"Task: {query_2}"},
            # Jika ada action-history sebelumnya, bisa ditambahkan di sini
            {"type": "image", "image": img_url_2, "min_pixels": min_pixels, "max_pixels": max_pixels},
        ],
    }
]

text_2 = processor.apply_chat_template(
    messages_2, tokenize=False, add_generation_prompt=True,
)
image_inputs_2, video_inputs_2 = process_vision_info(messages_2)
inputs_2 = processor(
    text=[text_2],
    images=image_inputs_2,
    videos=video_inputs_2,
    padding=True,
    return_tensors="pt",
)
inputs_2 = inputs_2.to(model.device)

start_time_2 = time.time()
generated_ids_2 = model.generate(**inputs_2, max_new_tokens=128)
end_time_2 = time.time()

# Hitung waktu inferensi UI Navigation
time_inference_2 = end_time_2 - start_time_2
print(f"Inference Time for UI Navigation: {time_inference_2:.2f} seconds.")

generated_ids_trimmed_2 = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_2.input_ids, generated_ids_2)
]
output_text_2 = processor.batch_decode(
    generated_ids_trimmed_2, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("\n=== Output Aksi Navigasi ===")
print(output_text_2)
# Contoh output:
# {'action': 'CLICK', 'value': None, 'position': [0.49, 0.42]},
# {'action': 'INPUT', 'value': 'weather for New York city', 'position': [0.49, 0.42]},
# {'action': 'ENTER', 'value': None, 'position': None}

# Untuk menampilkan gambar kedua
image_2 = Image.open(img_url_2)
image_2.show()
