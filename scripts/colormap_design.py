#!/usr/bin/env python3
"""
Colormap design iteration script for sledding scores.

Creates visual examples for iterating on custom colormap design.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import seaborn as sns


def show_colormap_comparison(output_dir: Path):
    """Generate comparison images of mako vs proposed custom colormaps."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sample score data with various patterns
    np.random.seed(42)

    # Pattern 1: Gradient with some noise
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    gradient_data = X + np.random.rand(200, 200) * 0.1
    gradient_data = np.clip(gradient_data, 0, 1)

    # Pattern 2: Circular "hills" of high scores (simulates good sledding areas)
    hills_data = np.zeros((200, 200))
    # Add several circular high-score regions
    centers = [(50, 50), (150, 100), (80, 150), (160, 40)]
    for cx, cy in centers:
        yy, xx = np.ogrid[:200, :200]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        hills_data += np.clip(1 - dist/40, 0, 1)
    hills_data = np.clip(hills_data, 0, 1)

    # Pattern 3: Real-ish sledding pattern (some noise + hills)
    realistic_data = hills_data * 0.7 + np.random.rand(200, 200) * 0.3
    realistic_data = np.clip(realistic_data, 0, 1)

    # Get reference mako colormap from seaborn
    mako = sns.color_palette("mako", as_cmap=True)

    # Sample mako colors at key positions
    print("\nMako colormap key colors:")
    for pos in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rgb = mako(pos)[:3]
        print(f"  {pos:.2f}: RGB({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})")

    # Create Figure 1: Mako reference
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Reference: Mako Colormap (viridis family)', fontsize=14)

    for ax, (data, title) in zip(axes, [
        (gradient_data, 'Gradient'),
        (hills_data, 'Hills Pattern'),
        (realistic_data, 'Realistic Sledding')
    ]):
        im = ax.imshow(data, cmap=mako, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8, label='Score')
    plt.tight_layout()
    plt.savefig(output_dir / '01_mako_reference.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / '01_mako_reference.png'}")

    return gradient_data, hills_data, realistic_data


def create_boreal_mako_v1(output_dir: Path, gradient_data, hills_data, realistic_data):
    """
    Version 1: Boreal Mako - Replace low end with dark green.

    Mako goes: dark purple/blue -> teal -> bright cyan/yellow
    This version: dark boreal green -> teal -> bright cyan/yellow
    """
    # Custom colormap: boreal green at low, keep mako's mid-high colors
    colors_v1 = [
        (0.0, (0.05, 0.15, 0.08)),   # Dark boreal green (almost black-green)
        (0.2, (0.08, 0.25, 0.12)),   # Deep forest green
        (0.4, (0.15, 0.40, 0.35)),   # Transition to teal
        (0.6, (0.20, 0.55, 0.50)),   # Mako's teal zone
        (0.8, (0.45, 0.75, 0.65)),   # Mako's light teal
        (1.0, (0.85, 0.95, 0.65)),   # Mako's bright end (slightly green-yellow)
    ]

    cmap_v1 = LinearSegmentedColormap.from_list('boreal_mako_v1',
        [c[1] for c in colors_v1], N=256)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('V1: Boreal Mako - Dark Green Low End', fontsize=14)

    for ax, (data, title) in zip(axes, [
        (gradient_data, 'Gradient'),
        (hills_data, 'Hills Pattern'),
        (realistic_data, 'Realistic Sledding')
    ]):
        im = ax.imshow(data, cmap=cmap_v1, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8, label='Score')
    plt.tight_layout()
    plt.savefig(output_dir / '02_boreal_mako_v1.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / '02_boreal_mako_v1.png'}")

    return cmap_v1


def create_boreal_mako_v2_sharp(output_dir: Path, gradient_data, hills_data, realistic_data):
    """
    Version 2: Boreal Mako with SHARP transition at high scores.

    Creates an "edge" effect by having colors change slowly in low-mid range,
    then jump dramatically at high scores.
    """
    # Sharp transition: slow change 0-0.7, rapid change 0.7-1.0
    colors_v2 = [
        (0.00, (0.05, 0.15, 0.08)),   # Dark boreal green
        (0.20, (0.08, 0.22, 0.12)),   # Slightly lighter forest
        (0.40, (0.12, 0.30, 0.18)),   # Still quite dark green
        (0.60, (0.18, 0.40, 0.28)),   # Medium forest green
        (0.70, (0.22, 0.48, 0.35)),   # Transitioning...
        (0.80, (0.55, 0.75, 0.45)),   # JUMP! Bright lime-green
        (0.90, (0.80, 0.90, 0.55)),   # Very bright yellow-green
        (1.00, (0.95, 0.98, 0.70)),   # Near white/cream highlight
    ]

    cmap_v2 = LinearSegmentedColormap.from_list('boreal_mako_v2',
        [c[1] for c in colors_v2], N=256)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('V2: Sharp Transition at 0.7 (Edge Effect)', fontsize=14)

    for ax, (data, title) in zip(axes, [
        (gradient_data, 'Gradient'),
        (hills_data, 'Hills Pattern'),
        (realistic_data, 'Realistic Sledding')
    ]):
        im = ax.imshow(data, cmap=cmap_v2, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8, label='Score')
    plt.tight_layout()
    plt.savefig(output_dir / '03_boreal_sharp_v2.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / '03_boreal_sharp_v2.png'}")

    return cmap_v2


def create_boreal_mako_v3_sharper(output_dir: Path, gradient_data, hills_data, realistic_data):
    """
    Version 3: Even sharper transition for more dramatic edge effect.

    Very slow change from 0-0.8, dramatic jump at 0.8+
    """
    colors_v3 = [
        (0.00, (0.03, 0.12, 0.05)),   # Very dark boreal (near black)
        (0.30, (0.06, 0.18, 0.10)),   # Still very dark
        (0.50, (0.10, 0.25, 0.15)),   # Dark forest green
        (0.70, (0.15, 0.35, 0.22)),   # Medium-dark green
        (0.80, (0.20, 0.42, 0.28)),   # Transition point
        (0.85, (0.60, 0.78, 0.40)),   # SHARP JUMP to bright
        (0.92, (0.85, 0.92, 0.55)),   # Very bright
        (1.00, (1.00, 1.00, 0.75)),   # White-gold peak
    ]

    cmap_v3 = LinearSegmentedColormap.from_list('boreal_mako_v3',
        [c[1] for c in colors_v3], N=256)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('V3: Very Sharp Transition at 0.8 (Strong Edge)', fontsize=14)

    for ax, (data, title) in zip(axes, [
        (gradient_data, 'Gradient'),
        (hills_data, 'Hills Pattern'),
        (realistic_data, 'Realistic Sledding')
    ]):
        im = ax.imshow(data, cmap=cmap_v3, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8, label='Score')
    plt.tight_layout()
    plt.savefig(output_dir / '04_boreal_sharper_v3.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / '04_boreal_sharper_v3.png'}")

    return cmap_v3


def create_boreal_mako_v4_blended(output_dir: Path, gradient_data, hills_data, realistic_data):
    """
    Version 4: Boreal green fading INTO mako's blue, then compressed high end.

    0.0-0.15: Dark boreal green fading to mako's dark purple-blue
    0.15-0.7: Follow mako's natural progression (purple → teal → cyan)
    0.7-1.0: Compressed bright transition for edge/outline effect
    """
    # Mako reference points (from seaborn):
    # 0.00: (0.045, 0.015, 0.021) - near black
    # 0.25: (0.244, 0.207, 0.420) - deep purple
    # 0.50: (0.207, 0.482, 0.638) - teal blue
    # 0.75: (0.292, 0.761, 0.679) - bright cyan-green
    # 1.00: (0.872, 0.960, 0.897) - pale mint

    colors_v4 = [
        # Boreal start -> fade to mako's blue
        (0.00, (0.02, 0.08, 0.04)),    # Very dark boreal green
        (0.08, (0.04, 0.06, 0.08)),    # Transitioning to blue-ish
        (0.15, (0.10, 0.08, 0.18)),    # Into mako's dark purple zone

        # Follow mako's progression
        (0.25, (0.20, 0.18, 0.38)),    # Mako's purple (slightly adjusted)
        (0.40, (0.18, 0.38, 0.55)),    # Mako's blue-teal transition
        (0.55, (0.20, 0.52, 0.64)),    # Mako's teal
        (0.70, (0.28, 0.72, 0.68)),    # Mako's cyan-green

        # Compressed bright end for edge effect
        (0.80, (0.50, 0.85, 0.75)),    # Brighter cyan
        (0.90, (0.75, 0.94, 0.85)),    # Very bright
        (1.00, (0.92, 0.98, 0.92)),    # Near-white mint peak
    ]

    cmap_v4 = LinearSegmentedColormap.from_list('boreal_mako_v4',
        [c[1] for c in colors_v4], N=256)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('V4: Boreal → Mako Blue → Compressed Bright End', fontsize=14)

    for ax, (data, title) in zip(axes, [
        (gradient_data, 'Gradient'),
        (hills_data, 'Hills Pattern'),
        (realistic_data, 'Realistic Sledding')
    ]):
        im = ax.imshow(data, cmap=cmap_v4, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8, label='Score')
    plt.tight_layout()
    plt.savefig(output_dir / '05_boreal_mako_blended_v4.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / '05_boreal_mako_blended_v4.png'}")

    return cmap_v4


def create_boreal_mako_v5_sharper_edge(output_dir: Path, gradient_data, hills_data, realistic_data):
    """
    Version 5: Same as V4 but with SHARPER edge transition at 0.75+

    More dramatic outline effect - stays dark longer, then pops bright.
    """
    colors_v5 = [
        # Boreal start -> fade to mako's blue
        (0.00, (0.02, 0.08, 0.04)),    # Very dark boreal green
        (0.10, (0.05, 0.06, 0.12)),    # Transitioning to blue
        (0.18, (0.12, 0.10, 0.22)),    # Into mako's dark purple

        # Follow mako but slower/darker
        (0.30, (0.18, 0.16, 0.35)),    # Mako's purple
        (0.45, (0.18, 0.35, 0.50)),    # Blue-teal
        (0.60, (0.20, 0.48, 0.60)),    # Teal
        (0.75, (0.25, 0.62, 0.65)),    # Cyan-green (held back)

        # SHARP edge transition
        (0.82, (0.45, 0.80, 0.72)),    # Jump starts
        (0.88, (0.70, 0.92, 0.82)),    # Bright
        (0.94, (0.88, 0.97, 0.90)),    # Very bright
        (1.00, (0.98, 1.00, 0.95)),    # Near-white peak
    ]

    cmap_v5 = LinearSegmentedColormap.from_list('boreal_mako_v5',
        [c[1] for c in colors_v5], N=256)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('V5: Boreal → Mako → SHARP Edge at 0.82', fontsize=14)

    for ax, (data, title) in zip(axes, [
        (gradient_data, 'Gradient'),
        (hills_data, 'Hills Pattern'),
        (realistic_data, 'Realistic Sledding')
    ]):
        im = ax.imshow(data, cmap=cmap_v5, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8, label='Score')
    plt.tight_layout()
    plt.savefig(output_dir / '06_boreal_mako_sharp_edge_v5.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / '06_boreal_mako_sharp_edge_v5.png'}")

    return cmap_v5


def create_boreal_mako_v8_bright_boreal(output_dir: Path, gradient_data, hills_data, realistic_data):
    """
    Version 8: Brighter boreal green matching mako blue's luminance.

    Mako blue at 0.25 has RGB (0.244, 0.207, 0.420), luminance ≈ 0.24
    We want boreal green with similar perceived brightness.

    Luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    For (0.08, 0.35, 0.12): Y = 0.024 + 0.205 + 0.014 = 0.243 ✓

    0.0-0.08: Deep Boreal (dark)
    0.08-0.22: BRIGHT Boreal green (matches mako blue luminance)
    0.22-0.50: Transition to mako blue-teal
    0.50-0.60: SHARP compressed transition (edge effect)
    0.60-1.0: Expanded pale green → mint
    """
    colors_v8 = [
        # Deep boreal base (still dark)
        (0.00, (0.04, 0.10, 0.06)),    # Deep Boreal (Y=0.07)
        (0.06, (0.05, 0.14, 0.08)),    # Black Spruce (Y=0.10)

        # BRIGHT boreal - matches mako blue luminance (~0.24)
        (0.12, (0.08, 0.35, 0.12)),    # Lively Boreal (Y=0.24) - matches mako!
        (0.18, (0.10, 0.38, 0.14)),    # Bright Forest (Y=0.26)

        # Transition to mako (green → blue-purple → teal)
        (0.24, (0.14, 0.30, 0.22)),    # Green fading to blue undertone
        (0.30, (0.20, 0.22, 0.36)),    # Into mako purple
        (0.38, (0.24, 0.28, 0.48)),    # Mako blue-purple
        (0.44, (0.22, 0.40, 0.58)),    # Mako blue
        (0.50, (0.21, 0.48, 0.64)),    # Mako teal

        # COMPRESSED edge transition [0.50-0.60]
        (0.55, (0.29, 0.70, 0.68)),    # Jump to bright cyan
        (0.60, (0.50, 0.82, 0.70)),    # Pale green

        # Expanded pale [0.60-1.0]
        (0.70, (0.64, 0.87, 0.75)),
        (0.80, (0.76, 0.91, 0.82)),
        (0.90, (0.84, 0.94, 0.87)),
        (1.00, (0.87, 0.96, 0.90)),    # Mint
    ]

    cmap_v8 = LinearSegmentedColormap.from_list('boreal_mako_v8',
        [c[1] for c in colors_v8], N=256)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('V8: Deep Boreal → BRIGHT Boreal → Mako → SHARP → Pale', fontsize=14)

    for ax, (data, title) in zip(axes, [
        (gradient_data, 'Gradient'),
        (hills_data, 'Hills Pattern'),
        (realistic_data, 'Realistic Sledding')
    ]):
        im = ax.imshow(data, cmap=cmap_v8, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8, label='Score')
    plt.tight_layout()
    plt.savefig(output_dir / '09_boreal_mako_v8_bright.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / '09_boreal_mako_v8_bright.png'}")

    # Print luminance comparison
    print("\nLuminance comparison (V8 vs Mako):")
    mako = sns.color_palette("mako", as_cmap=True)
    for pos, rgb in colors_v8[:6]:
        Y = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        mako_rgb = mako(pos)[:3]
        mako_Y = 0.299*mako_rgb[0] + 0.587*mako_rgb[1] + 0.114*mako_rgb[2]
        print(f"  {pos:.2f}: V8 Y={Y:.3f}, Mako Y={mako_Y:.3f}")

    return cmap_v8


def create_comparison_strip(output_dir: Path, cmaps: dict):
    """Create a side-by-side colorbar comparison."""
    fig, axes = plt.subplots(len(cmaps), 1, figsize=(10, len(cmaps) * 0.8))
    fig.suptitle('Colormap Comparison', fontsize=14, y=0.98)

    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    for ax, (name, cmap) in zip(axes, cmaps.items()):
        ax.imshow(gradient, cmap=cmap, aspect='auto')
        ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=10)
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        ax.set_yticks([])

    axes[-1].set_xlabel('Score Value')
    plt.tight_layout()
    plt.savefig(output_dir / '05_comparison_strip.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / '05_comparison_strip.png'}")


def main():
    output_dir = Path('docs/images/colormap_design')

    print("=" * 60)
    print("Sledding Colormap Design - Iteration Examples")
    print("=" * 60)

    # Generate reference and test data
    gradient_data, hills_data, realistic_data = show_colormap_comparison(output_dir)

    # Create versions
    cmap_v1 = create_boreal_mako_v1(output_dir, gradient_data, hills_data, realistic_data)
    cmap_v2 = create_boreal_mako_v2_sharp(output_dir, gradient_data, hills_data, realistic_data)
    cmap_v3 = create_boreal_mako_v3_sharper(output_dir, gradient_data, hills_data, realistic_data)
    cmap_v4 = create_boreal_mako_v4_blended(output_dir, gradient_data, hills_data, realistic_data)
    cmap_v5 = create_boreal_mako_v5_sharper_edge(output_dir, gradient_data, hills_data, realistic_data)

    # Comparison strip
    create_comparison_strip(output_dir, {
        'mako (reference)': sns.color_palette("mako", as_cmap=True),
        'V4: Boreal→Mako': cmap_v4,
        'V5: Sharp Edge': cmap_v5,
    })

    print("\n" + "=" * 60)
    print("Images saved to:", output_dir)
    print("=" * 60)
    print("\nReview the images and let me know:")
    print("  1. Is the dark green (boreal) tone right?")
    print("  2. Is the transition sharp enough? Where should it occur?")
    print("  3. What color should the high scores be? (current: bright yellow-green)")


if __name__ == '__main__':
    main()
