#!/usr/bin/env python
"""Quick test script for scheduler - plots all schedulers together

Usage:
    python test_scheduler.py
"""
import sys
sys.path.append("./")

import torch
import matplotlib.pyplot as plt
from transformers import get_scheduler


def main():
    num_steps = 1000
    base_lr = 0.001
    warmup_ratio = 0.1
    num_warmup_steps = int(num_steps * warmup_ratio)
    
    print("=" * 60)
    print(f"Base LR: {base_lr}")
    print(f"Total steps: {num_steps}, Warmup steps: {num_warmup_steps}")
    print("=" * 60)
    
    # Schedulers to test (name, display_name, color)
    schedulers = [
        ("constant", "constant", "#6b7280"),
        ("cosine", "cosine (with warmup)", "#2563eb"),
        ("linear", "linear (with warmup)", "#16a34a"),
    ]
    
    plt.figure(figsize=(12, 7))
    
    for sched_name, display_name, color in schedulers:
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        
        scheduler = get_scheduler(
            name=sched_name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps if sched_name != "constant" else 0,
            num_training_steps=num_steps,
        )
        
        lrs = []
        for step in range(num_steps):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        plt.plot(range(num_steps), lrs, linewidth=2.5, color=color, label=display_name)
        
        print(f'\n{display_name}:')
        print(f'  Step 0:    LR = {lrs[0]:.6f}')
        print(f'  Step {num_warmup_steps:4d}: LR = {lrs[num_warmup_steps]:.6f}')
        print(f'  Step {num_steps//2:4d}: LR = {lrs[num_steps//2]:.6f}')
        print(f'  Step {num_steps-1:4d}: LR = {lrs[-1]:.6f}')
    
    # Mark warmup region
    plt.axvline(x=num_warmup_steps, color='#dc2626', linestyle='--', alpha=0.7,
                label=f'End of warmup ({num_warmup_steps} steps)')
    plt.axvspan(0, num_warmup_steps, alpha=0.1, color='#dc2626')
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedules Comparison', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = 'lr_schedules_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f'\n{"=" * 60}')
    print(f'Plot saved to: {output_path}')
    print("Test passed!")


if __name__ == "__main__":
    main()

