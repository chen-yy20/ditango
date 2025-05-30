import os
import time
import torch

from diffusers.utils import export_to_video
# from diffusers import CogVideoXPipeline

from ditango.core.config import get_config
from ditango.core.initialize import init_ditango
from ditango.executor.cogvideox import CogVideoXPipeline
from ditango.executor.cogvideox import CogVideoXTransformer3DModel

from ditango.logger import init_logger
from ditango.timer import print_time_statistics
from ditango.baseline.cache import clear_cache

logger = init_logger(__name__)

# reset CUDA memory
torch.cuda.reset_max_memory_allocated()

init_ditango(config_path="./ditango/configs/cogvideox-5b/config.yaml")
config = get_config()

generator = torch.Generator().manual_seed(config.seed)
device = torch.device("cuda", config.local_rank)
model_path = "./ckpts"
pipe = CogVideoXPipeline.from_pretrained(
  model_path,
  torch_dtype=torch.bfloat16,
  transformer=CogVideoXTransformer3DModel.from_pretrained(os.path.join(model_path, "transformer"), torch_dtype=torch.bfloat16),
)
pipe = pipe.to(device)
print(f"Finished loading pipeline", flush=True)

height = 480
width = 720
frames = 49
fps = 8

save_path = config.output_dir
prompts = [
  # "A video of a cat playing with a ball",
  "A playful black Labrador, adorned in a vibrant pumpkin-themed Halloween costume, frolics in a sunlit autumn garden, surrounded by fallen leaves. The dog's costume features a bright orange body with a green leafy collar, perfectly complementing its shiny black fur. As it bounds joyfully across the lawn, the sunlight catches the costume's fabric, creating a delightful contrast with the dog's dark coat. The scene captures the essence of autumn festivities, with the dog's wagging tail and playful demeanor adding to the cheerful atmosphere. Nearby, carved pumpkins and scattered leaves enhance the festive setting.",
  # "A charming boat glides gracefully along the serene Seine River, its sails catching a gentle breeze, while the iconic Eiffel Tower stands majestically in the background. The scene is rendered in rich, textured oil paints, capturing the warm hues of a late afternoon sun casting a golden glow over the water. The boat, with its elegant design and vibrant colors, contrasts beautifully with the soft, impressionistic strokes of the surrounding landscape. The Eiffel Tower, painted in delicate detail, rises above the Parisian skyline, its iron latticework shimmering in the light. The riverbanks are adorned with lush greenery and quaint buildings, their reflections dancing on the water's surface, creating a harmonious blend of nature and architecture. The overall composition exudes a sense of tranquility and timeless beauty, inviting viewers to immerse themselves in the idyllic Parisian scene.",
  # "A cat walks on the grass, realistic style.",
  # "A little girl is riding a bicycle at high speed. Focused, detailed, realistic.",
  # "Sun set over the sea"
]

if config.output_fn != "none":
    os.makedirs(config.output_fn, exist_ok=True)
    for id, prompt in enumerate(prompts):
        if config.local_rank == 0:
            print(f"{id+1} - {prompt}\n Begin generation...", flush=True)
        video = pipe(
            height=height,
            width=width,
            num_frames=frames,
            prompt=prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=6,
            generator=generator,
            ).frames[0]
        # Clear ditango cache
        clear_cache()
        if config.rank == 0:
            video_name = f"{id+1}_Cog5B_{prompt.split()[1]}_{config.tag}.mp4" # .mp4 if full video
            video_output_path = os.path.join(save_path, video_name)
            export_to_video(video, video_output_path, fps=fps)
            # image_output_path = os.path.join(config.output_fn, f"Cog5B_{prompt.split()[1]}_{tag}.png")
            # image = video[5]
            # image.save(image_output_path)
            logger.info(f"'{prompt} - saved to {video_output_path}!'")
        
print_time_statistics()
max_memory = torch.cuda.max_memory_allocated() 
print(f'最大显存使用: {max_memory / 1024**2:.2f} MB')
torch.distributed.destroy_process_group()



