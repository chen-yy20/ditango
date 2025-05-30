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
    )
    config = get_config()
    assert config.do_preprocess, "Set 'do-preprocess = True' in config.yaml first."
    
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
    redundancy_preprocess(func, func_args)
    
if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()
    main()
    print_time_statistics()
    logger.info(f"Completed generating video for the prompt")
    max_memory = torch.cuda.max_memory_allocated() 
    print(f'Maximum GPU memory used: {max_memory / 1024**2:.2f} MB')