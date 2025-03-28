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
from ditango.timer import print_time_statistics, enable_timing
from ditango.core.stride_map import preprocess_for_stridemap

logger = init_logger(__name__)

init_ditango(
    config_path="./ditango/configs/cogvideox1.5-5b/config.yaml",
    use_timer=False,
)

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

# preprocess_for_stridemap(pipe)
# exit()

prompt_list = [
  # "A video of a cat playing with a ball",
  # "A playful black Labrador, adorned in a vibrant pumpkin-themed Halloween costume, frolics in a sunlit autumn garden, surrounded by fallen leaves. The dog's costume features a bright orange body with a green leafy collar, perfectly complementing its shiny black fur. As it bounds joyfully across the lawn, the sunlight catches the costume's fabric, creating a delightful contrast with the dog's dark coat. The scene captures the essence of autumn festivities, with the dog's wagging tail and playful demeanor adding to the cheerful atmosphere. Nearby, carved pumpkins and scattered leaves enhance the festive setting.",
  "A charming boat glides gracefully along the serene Seine River, its sails catching a gentle breeze, while the iconic Eiffel Tower stands majestically in the background. The scene is rendered in rich, textured oil paints, capturing the warm hues of a late afternoon sun casting a golden glow over the water. The boat, with its elegant design and vibrant colors, contrasts beautifully with the soft, impressionistic strokes of the surrounding landscape. The Eiffel Tower, painted in delicate detail, rises above the Parisian skyline, its iron latticework shimmering in the light. The riverbanks are adorned with lush greenery and quaint buildings, their reflections dancing on the water's surface, creating a harmonious blend of nature and architecture. The overall composition exudes a sense of tranquility and timeless beauty, inviting viewers to immerse themselves in the idyllic Parisian scene.",
  # "A cat walks on the grass, realistic style.",
  # "A little girl is riding a bicycle at high speed. Focused, detailed, realistic.",
  # "Sun set over the sea"
]

height = 480
width = 720
frames = 48
# pipe.vae.enable_tiling() 


if config.output_fn != "none":
    os.makedirs(config.output_fn, exist_ok=True)
    for prompt in prompt_list:
        if config.local_rank == 0:
            print(f"output_fn: {config.output_fn}\n prompt: {prompt}\n Begin generation...", flush=True)
        video = pipe(
            height=height,
            width=width,
            num_frames=frames,
            prompt=prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=6,
            generator=generator,
            ).frames[0]
        if config.rank == 0:
            video_name = f"Cog5B_{prompt.split()[1]}_{config.tag}.mp4" # .mp4 if full video
            video_output_path = os.path.join(config.output_fn, video_name)
            export_to_video(video, video_output_path, fps=8)
            # image_output_path = os.path.join(config.output_fn, f"Cog5B_{prompt.split()[1]}_{tag}.png")
            # image = video[5]
            # image.save(image_output_path)
            logger.info(f"'{prompt} - saved to {video_output_path}!'")
        
print_time_statistics()
torch.distributed.destroy_process_group()

        

    # exit()
    
    # enable_timing()
    # logger.info("Warmup")
    # for _ in range(config.warmup):
    #   _ = pipe(
    #     height=height,
    #     width=width,
    #     num_frames=frames,
    #     prompt=prompt,
    #     num_inference_steps=config.num_inference_steps,
    #     guidance_scale=6,
    #     generator=generator,
    #   ).frames[0]
    # logger.info("Warmup done")
    # clear_cache()

    # get_world_group().barrier()
    # torch.cuda.synchronize()

    # # continue
    # # exit()
    # torch.cuda.reset_peak_memory_stats()
    # # enable_timing()
    # li = []
    # for _ in range(config.repeat):
    #   tik = time.time()
    #   _ = pipe(
    #     height=height,
    #     width=width,
    #     num_frames=frames,
    #     prompt=prompt,
    #     # num_inference_steps=4,
    #     num_inference_steps=config.num_inference_steps,
    #     guidance_scale=6,
    #     generator=generator,
    #   ).frames[0]
    #   tok = time.time()
    #   li.append(tok - tik)
    #   logger.info(f"Total time: {li[-1]}")
    # if len(li) != 0:
    #   logger.info(f"{pipe.device=}")
    #   avg_sec = sum(li) / len(li)
    #   logger.info(f"{config.local_rank=}, {li=}, Time: {avg_sec:.3f} s")
    # max_memory = torch.cuda.max_memory_allocated() 
    # print(f'最大显存使用: {max_memory / 1024**2:.2f} MB')



