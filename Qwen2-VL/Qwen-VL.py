from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.image_utils import load_image  # Pastikan ini diimpor jika diperlukan
import torch
import time  # Untuk mengukur waktu inferensi
from PIL import Image  # Untuk memuat dan menampilkan gambar

torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

query = tokenizer.from_list_format([
    {'image': 'https://image.popmama.com/content-images/post/20220630/kucing-artis-korea-8-23ec76f91276556c4873fa797b2ed618.png?width=600&height=315'},# https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': 'Generate the caption in English with grounding:'},
])
inputs = tokenizer(query, return_tensors='pt')
inputs = inputs.to(model.device)

# Mengukur waktu inferensi
start_time = time.perf_counter()

pred = model.generate(**inputs)
response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
print(response)
# <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>

# Menghentikan pengukuran waktu inferensi
end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Konversi waktu ke menit dan detik
minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60

print(f"\nWaktu Inferensi: {minutes} menit {seconds:.2f} detik")

image = tokenizer.draw_bbox_on_latest_picture(response)
if image:
    image.save('2.jpg')
    
    # Memuat kembali gambar yang telah disimpan menggunakan PIL
    pil_image = Image.open('2.jpg')
    pil_image.show()  # Menampilkan gambar menggunakan PIL
else:
    print("no box")
