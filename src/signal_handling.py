"""
Simple signal handling for SLURM job preemption.

Only two status values:
- "requeue": timeout preemption with checkpoint saved
- "finished": training completed successfully

Usage in training code:
    from signal_handling import should_terminate, save_and_requeue, mark_finished
    
    for batch in dataloader:
        if should_terminate():
            save_and_requeue(accelerator, output_path, metrics)
            return
        # ... training ...
    
    mark_finished(output_path)
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional


def should_terminate() -> bool:
    """
    Check if termination was requested (via SLURM file marker).
    
    Call this at the start of each training iteration.
    Returns True if the job should save checkpoint and exit.
    """
    marker_path = os.environ.get("SLURM_TERMINATE_MARKER")
    return marker_path is not None and os.path.exists(marker_path)


def save_and_requeue(
    accelerator: Any,
    output_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    safe_tensors: bool = True,
) -> None:
    """
    Save checkpoint and mark job for requeue.
    
    ALL processes must call this together (distributed collective).
    """
    import torch
    
    save_path = os.path.join(output_path, "states", "latest")
    
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
    
    accelerator.wait_for_everyone()
    accelerator.save_state(save_path, safe_serialization=safe_tensors)
    
    if accelerator.is_main_process:
        if metrics:
            torch.save(metrics, os.path.join(save_path, "metrics.pt"))
        Path(output_path, "status.txt").write_text("requeue")
    
    # Exit immediately - don't wait_for_everyone() which can hang during preemption
    import sys
    sys.exit(0)


def mark_finished(accelerator: Any, output_path: str) -> None:
    """Mark training as successfully completed."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        Path(output_path, "status.txt").write_text("finished")
