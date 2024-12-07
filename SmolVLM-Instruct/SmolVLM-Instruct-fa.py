import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import time  # Untuk mengukur waktu inferensi

# Menentukan perangkat (GPU jika tersedia)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image1_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
image2_url = "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg"
image1 = load_image(image1_url)
image2 = load_image(image2_url)

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": "Can you describe the two images?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    text=prompt,
    images=[image1, image2],
    return_tensors="pt"
)
inputs = inputs.to(DEVICE)

# Mengukur waktu inferensi
start_time = time.perf_counter()

# Generate outputs
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Konversi waktu ke menit dan detik
minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60

# Menampilkan hasil
print("Hasil Generasi Teks:")
print(generated_texts[0])
print(f"\nWaktu Inferensi: {minutes} menit {seconds:.2f} detik")

# Menampilkan gambar menggunakan PIL
print("\nMenampilkan gambar...")

# Menampilkan gambar pertama
image1.show(title="Gambar 1")

# Menampilkan gambar kedua
image2.show(title="Gambar 2")
