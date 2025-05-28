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
from ditango.core.stride_map import preprocess_for_stridemap


from ditango.logger import init_logger
from ditango.timer import print_time_statistics
from ditango.executor.hunyuanvideo.inference import HunyuanVideoSampler

logger = init_logger(__name__)
 
def main():
    args = parse_args()
    init_ditango(config_path="/home/zhongrx/cyy/HunyuanVideo/ditango/configs/hunyuanvideo/config.yaml")
    config = get_config()
    
    # Define your specific prompt here
    prompt = "A kitten wearing a red bow tie is dancing, the camera slowly zooms in from a distance"
    if config.rank == 0:
        logger.info(f"Using prompt: {prompt[:100]}...")
    
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    # save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path, exist_ok=True)
    save_path = config.output_dir

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    logger.info("Finished building model, begin generate...")
    
    # Get the updated args
    args = hunyuan_video_sampler.args
    
    # Start sampling
    outputs = hunyuan_video_sampler.predict(
        prompt=prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    if config.rank == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            # Create a sanitized prompt for filename
            safe_prompt = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in prompt[:20])
            safe_prompt = safe_prompt.replace(' ', '_')
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
            save_file = f"{save_path}/{safe_prompt}_{config.tag}_{time_flag}.mp4"
            save_videos_grid(sample, save_file, fps=24)
            logger.info(f'Sample saved to: {save_file}')
        

if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()
    main()
    print_time_statistics()
    logger.info(f"Completed generating video for the prompt")
    max_memory = torch.cuda.max_memory_allocated() 
    print(f'Maximum GPU memory used: {max_memory / 1024**2:.2f} MB')