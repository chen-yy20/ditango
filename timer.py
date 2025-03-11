import time
import torch
from typing import Dict, Optional
from contextlib import ContextDecorator

class _TimerCore:
    """Internal timer implementation"""
    def __init__(self, name: str):
        self.name = name
        self.reset()
        
    def reset(self):
        """Reset all recorded times"""
        self.times = []
        self._start_time = 0

    def start(self):
        """Record start time"""
        torch.cuda.synchronize()  # 确保之前的GPU操作完成
        self._start_time = time.perf_counter()

    def stop(self):
        """Record end time and calculate duration"""
        torch.cuda.synchronize()  # 确保GPU操作完成
        elapsed_time = (time.perf_counter() - self._start_time) * 1000  # 转换为毫秒
        self.times.append(elapsed_time)

    @property
    def average(self) -> float:
        """Get average time in milliseconds"""
        return sum(self.times) / len(self.times) if self.times else 0.0

class TimerManager:
    """Global timer manager with control flags"""
    _instance = None
    _timers: Dict[str, _TimerCore] = {}
    _global_enabled = False

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(TimerManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def init_timer(cls, enable: bool = True):
        """Initialize global timer settings
        
        Args:
            enable: Global enable/disable flag for all timers
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but required for timing")
        cls._global_enabled = enable
        print(f"Timer initialized: enable={enable}")

    @classmethod
    def get_timer(cls, name: str) -> ContextDecorator:
        """Get or create a named timer
        
        Args:
            name: Unique identifier for the timer
        """
        if name not in cls._timers:
            cls._timers[name] = _TimerCore(name)
            
        class _TimerContext(ContextDecorator):
            """Context manager wrapper for timer"""
            def __enter__(self_ctx):
                if cls._global_enabled:
                    cls._timers[name].start()
                return self_ctx

            def __exit__(self_ctx, *args):
                if cls._global_enabled:
                    cls._timers[name].stop()

        return _TimerContext()

    @classmethod
    def print_time_statistics(cls):
        """Print timing statistics (min/max/avg) for all timers"""
        if not cls._global_enabled:
            print("Timing is disabled")
            return

        print("\n===== Timing Statistics =====")
        header_format = "{:<20} | {:>10} | {:>10} | {:>10} | {:>8}"
        print(header_format.format("Timer Name", "Min (ms)", "Max (ms)", "Avg (ms)", "Samples"))
        print("-" * 66)
        
        data_format = "{:<20} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>8d}"
        for name, timer in cls._timers.items():
            if timer.times:
                avg = sum(timer.times) / len(timer.times)
                min_time = min(timer.times)
                max_time = max(timer.times)
                print(data_format.format(
                    name[:20], 
                    min_time,
                    max_time,
                    avg,
                    len(timer.times)
                ))
        print("=" * 66 + "\n")
        
    @classmethod
    def enable_timing(cls):
        """Enable timing for all subsequent operations"""
        cls._global_enabled = True

    @classmethod
    def disable_timing(cls):
        """Disable timing for all subsequent operations"""
        cls._global_enabled = False

    @classmethod
    def is_timing_enabled(cls) -> bool:
        """Check if timing is currently enabled"""
        return cls._global_enabled

# Public interface functions
def init_timer(enable: bool = True):
    TimerManager.init_timer(enable)

def get_timer(name: str) -> ContextDecorator:
    return TimerManager.get_timer(name)

def print_time_statistics():
    TimerManager.print_time_statistics()
    
def enable_timing():
    """Globally enable timing measurement"""
    TimerManager.enable_timing()

def disable_timing():
    """Globally disable timing measurement"""
    TimerManager.disable_timing()

def is_timing_enabled() -> bool:
    """Check global timing enable status"""
    return TimerManager.is_timing_enabled()