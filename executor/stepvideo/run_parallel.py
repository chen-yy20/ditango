# from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch.distributed as dist
import torch
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed

from ditango.logger import init_logger
from ditango.core.initialize import init_ditango
from ditango.core.config import get_config
from ditango.timer import print_time_statistics
from ditango.executor.stepvideo.video_pipeline import StepVideoPipeline

logger = init_logger(__name__)

if __name__ == "__main__":
    args = parse_args()
    init_ditango(
        config_path="./ditango/configs/stepvideo/config.yaml",
        use_timer=True
    )
    device = torch.device(f"cuda:{get_config().local_rank}")
    
    setup_seed(args.seed)
        
    pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(dtype=torch.bfloat16, device=device)
    logger.info("Pipeline Loaded.")
    pipeline.setup_api(
        vae_url = args.vae_url,
        caption_url = args.caption_url,
    )
    logger.info("Pipeline set api.")
    
    
    prompt = args.prompt
    prompt = "橘色小猫带着红色领结，在舞台上跳出一个完美的旋转动作。"
    videos = pipeline(
        prompt=prompt, 
        num_frames=args.num_frames, 
        height=args.height, 
        width=args.width,
        num_inference_steps = args.infer_steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=prompt[:50]
    )
    
    dist.destroy_process_group()
    print_time_statistics()