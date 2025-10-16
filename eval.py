import torch
from prior_depth_anything import PriorDepthAnything
from PIL import Image
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
priorda = PriorDepthAnything(device=device)
image_path = '/users/aidhant/data/aidhant/color/color_test_data1210.png'
image_array = Image.open(image_path)
image_array = np.array(image_array, dtype=np.float32)
H, W = image_array.shape[:2]
print(f'This is the H {image_array.dtype} and W {W} of the incoming data')




prior_path = '/users/aidhant/data/aidhant/depth/depth_test_data1210.npy'
loaded_array = np.load(prior_path).astype(np.float32)
reshaped = loaded_array.reshape((H, W))
print(f'This is the H {loaded_array.dtype} ')


output = priorda.infer_one_sample(image=image_array, prior=reshaped, visualize=True)

