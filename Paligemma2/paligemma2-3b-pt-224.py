from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch
from PIL import Image
import time  # Import modul time untuk pengukuran waktu

model_id = "google/paligemma2-3b-pt-224"

# URL gambar yang akan diproses
# url = "https://dfcm824dmlg8u.cloudfront.net/wp-content/uploads/2023/04/pesawat-baling-baling-1.jpg"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = load_image(url)

# Memuat model dan processor
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
).eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Membuat prompt kosong untuk model pra-latih
prompt = ""
model_inputs = processor(
    text=prompt, 
    images=image, 
    return_tensors="pt"
).to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

# Mengukur waktu inferensi
start_time = time.perf_counter()  # Mulai pengukuran waktu

with torch.inference_mode():
    generation = model.generate(
        **model_inputs, 
        max_new_tokens=100, 
        do_sample=False
    )
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)

end_time = time.perf_counter()  # Selesai pengukuran waktu
elapsed_time = end_time - start_time  # Hitung waktu yang berlalu

# Konversi waktu ke menit dan detik
minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60

print("Hasil Generasi Teks:")
print(" ")
print(decoded)
print(" ")
print(f"Waktu Inferensi: {minutes} menit {seconds:.2f} detik")

# Menampilkan gambar menggunakan PIL
print("Menampilkan gambar...")
image.show()
