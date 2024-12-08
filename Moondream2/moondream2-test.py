from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time  # Untuk mengukur waktu inferensi

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

# use cuda device
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Memuat gambar dari file lokal
image = Image.open('api.png')

# Mengukur waktu inferensi
start_time = time.perf_counter()

# Encode gambar dan menjawab pertanyaan
enc_image = model.encode_image(image)
response = model.answer_question(enc_image, "Describe this image.", tokenizer)

# Menghentikan pengukuran waktu inferensi
end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Menampilkan respons dari model
print(response)

# Menampilkan waktu inferensi
minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60
print(f"\nWaktu Inferensi: {minutes} menit {seconds:.2f} detik")

# Menampilkan gambar menggunakan PIL
image.show()
