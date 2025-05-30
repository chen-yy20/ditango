import os
import time
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime

from hyvideo.config import parse_args
from hyvideo.utils.file_utils import save_videos_grid
# from hyvideo.inference import HunyuanVideoSampler

from ditango.core.initialize import init_ditango
from ditango.core.config import get_config
from ditango.core.redundancy_map import redundancy_preprocess


from ditango.logger import init_logger
from ditango.timer import print_time_statistics
from ditango.executor.hunyuanvideo.inference import HunyuanVideoSampler

logger = init_logger(__name__)
 
def main():
    args = parse_args()
    init_ditango(
        config_path="./ditango/configs/hunyuanvideo/config.yaml",
        use_timer=True,
    )
    config = get_config()
    
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    logger.info("Finished building model, begin generate...")
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    func = hunyuan_video_sampler.predict
    func_args = {
        "height": args.video_size[0],
        "width": args.video_size[1],
        "video_length": args.video_length,
        "seed": args.seed,
        "negative_prompt": args.neg_prompt,
        "infer_steps": args.infer_steps,
        "guidance_scale": args.cfg_scale,
        "num_videos_per_prompt": args.num_videos,
        "flow_shift": args.flow_shift,
        "batch_size": args.batch_size,
        "embedded_guidance_scale": args.embedded_cfg_scale
    }
    prompt_list = [
    "A playful black Labrador, adorned in a vibrant pumpkin-themed Halloween costume, frolics in a sunlit autumn garden, surrounded by fallen leaves. The dog's costume features a bright orange body with a green leafy collar, perfectly complementing its shiny black fur. As it bounds joyfully across the lawn, the sunlight catches the costume's fabric, creating a delightful contrast with the dog's dark coat. The scene captures the essence of autumn festivities, with the dog's wagging tail and playful demeanor adding to the cheerful atmosphere. Nearby, carved pumpkins and scattered leaves enhance the festive setting.",
    "A charming boat glides gracefully along the serene Seine River, its sails catching a gentle breeze, while the iconic Eiffel Tower stands majestically in the background. The scene is rendered in rich, textured oil paints, capturing the warm hues of a late afternoon sun casting a golden glow over the water. The boat, with its elegant design and vibrant colors, contrasts beautifully with the soft, impressionistic strokes of the surrounding landscape. The Eiffel Tower, painted in delicate detail, rises above the Parisian skyline, its iron latticework shimmering in the light. The riverbanks are adorned with lush greenery and quaint buildings, their reflections dancing on the water's surface, creating a harmonious blend of nature and architecture. The overall composition exudes a sense of tranquility and timeless beauty, inviting viewers to immerse themselves in the idyllic Parisian scene.",
    "A cat walks on the grass, realistic style.",
    ]
    redundancy_preprocess(func, func_args, prompt_list)
    
if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()
    main()
    print_time_statistics()
    logger.info(f"Completed generating video for the prompt")
    max_memory = torch.cuda.max_memory_allocated() 
    print(f'Maximum GPU memory used: {max_memory / 1024**2:.2f} MB')