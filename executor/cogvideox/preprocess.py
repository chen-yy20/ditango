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
from ditango.core.redundancy_map import redundancy_preprocess
from ditango.baseline.cache import clear_cache

logger = init_logger(__name__)

# reset CUDA memory
torch.cuda.reset_max_memory_allocated()

init_ditango(config_path="./ditango/configs/cogvideox-5b/config.yaml")
config = get_config()
assert config.do_preprocess, "Set 'do-preprocess = True' in config.yaml first."

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

func = pipe
func_args = {
    "height": height,
    "width": width,
    "num_frames": frames,
    "num_inference_steps": config.num_inference_steps,
    "guidance_scale": 6,
    "generator": generator,
}

redundancy_preprocess(func, func_args)

