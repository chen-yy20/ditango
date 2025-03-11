# arguments.py
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Video generation model arguments")
    
    # 基本参数
    parser.add_argument('--model-type', type=str, default='cogvideox',
                        choices=['cogvideox', 'mochi', 'opensora', 'hunyuan', 'latte'],
                        help='Model type to use')
    parser.add_argument('--path', type=str, default='THUDM/CogVideoX-5b',
                        help='Path to model checkpoint')
    parser.add_argument('--output-fn', '-o', type=str, default='output',
                        help='Output directory for generated videos')
    
    # 生成参数
    parser.add_argument('--prompt', type=str, default='Sunset over the sea.',
                        help='Text prompt for video generation')
    parser.add_argument('--prompt-list', type=str, nargs='+', 
                        help='List of prompts for batch generation')
    parser.add_argument('--height', type=int, default=480, 
                        help='Height of generated video')
    parser.add_argument('--width', type=int, default=720, 
                        help='Width of generated video')
    parser.add_argument('--frames', type=int, default=48, 
                        help='Number of frames to generate')
    parser.add_argument('--num-inference-steps', type=int, default=50, 
                        help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=6.0, 
                        help='Guidance scale for classifier-free guidance')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for generation')
    
    # 性能测试参数
    parser.add_argument('--warmup', type=int, default=0,
                        help='Number of warmup runs before benchmarking')
    parser.add_argument('--repeat', type=int, default=2,
                        help='Number of repeat runs for benchmarking')
    parser.add_argument('--enable-timing', action='store_true',
                        help='Enable detailed timing statistics')
    parser.add_argument('--tag', type=str, default='test',
                        help='Tag for the experiment')
    
    # 分布式参数
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to use')
    parser.add_argument('--node', type=str, default=None,
                        help='Node to use for distributed training')
    
    # 用于Baseline实验
    parser.add_argument('--use-ulysses', action='store_true',
                        help='evaluate ulysses baseline')
    parser.add_argument('--use-distrifusion', action='store_true',
                        help='evaluate ulysses baseline')
    
    # 缓存和优化参数
    parser.add_argument('--use-easy-cache', action='store_true',
                        help='evaluate easyCache baseline')
    parser.add_argument('--cache-threshold', type=int, default=4,
                        help='Threshold for feature caching')
    
    # 环境变量解析
    args = parser.parse_args()
    
    # 从环境变量获取分布式训练信息
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.rank = int(os.getenv("RANK", "0"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.tag = os.getenv("TAG", args.tag)
    
    # 如果未提供prompt_list但提供了prompt，使用单个prompt
    if args.prompt_list is None and args.prompt:
        args.prompt_list = [args.prompt]
    
    # 如果未提供prompt_list，使用默认prompt
    if args.prompt_list is None:
        args.prompt_list = [
            "A playful black Labrador, adorned in a vibrant pumpkin-themed Halloween costume, frolics in a sunlit autumn garden, surrounded by fallen leaves.",
            "A cat walks on the grass, realistic style.",
            "Sun set over the sea"
        ]
    
    return args

# 全局参数对象
args = None

def init_args():
    """初始化全局参数对象"""
    global args
    args = parse_args()
    return args

def get_args():
    """获取全局参数对象，如果未初始化则初始化"""
    global args
    if args is None:
        args = init_args()
    return args