"""
draw_architecture.py — Generate high-resolution publication-quality architecture diagram
for R3D-18 Multi-Task Action Units Micro-Expression Recognition (Run 188 Champion) in English.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Polygon

def create_architecture_diagram(output_path):
    fig = plt.figure(figsize=(23.5, 12.0), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 23.5)
    ax.set_ylim(0, 12.0)
    ax.axis('off')

    bg_color = "#f8fafc"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # -------------------------------------------------------------
    # HEADER SECTION
    # -------------------------------------------------------------
    ax.text(11.75, 11.5, "CASME II Champion Model Architecture: R3D-18 Multi-Task Facial Action Units (Run 188)",
            ha='center', va='center', fontsize=18.5, fontweight='bold', color="#0f172a")
    ax.text(11.75, 11.05, "TV-L1 Optical Flow Volume (16 Frames)  ➔  Spatiotemporal 3D ResNet-18 Backbone  ➔  Dual Emotion & FACS AU Heads",
            ha='center', va='center', fontsize=11.5, color="#475569")

    # Helper function for cards
    def draw_card(x, y, w, h, title="", subtitle="", color="#3b82f6", fill="#ffffff", rad=0.15, lw=1.8,
                  title_y_offset=0.25, subtitle_y_offset=0.55):
        shadow = FancyBboxPatch((x + 0.04, y - 0.04), w, h,
                                boxstyle=f"round,pad={rad}",
                                fc="#e2e8f0", ec="none", zorder=1)
        ax.add_patch(shadow)
        card = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad={rad}",
                              fc=fill, ec=color, lw=lw, zorder=2)
        ax.add_patch(card)
        if title:
            ax.text(x + w/2, y + h - title_y_offset, title, ha='center', va='center',
                    fontsize=10.5, fontweight='bold', color=color, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - subtitle_y_offset, subtitle, ha='center', va='center',
                    fontsize=8.5, color="#64748b", zorder=3)
        return card

    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2, color="#64748b", lw=2, text="", rad=0):
        if rad == 0:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                        mutation_scale=14, shrinkA=2, shrinkB=2),
                        zorder=4)
        else:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                        connectionstyle=f"arc3,rad={rad}",
                                        mutation_scale=14, shrinkA=2, shrinkB=2),
                        zorder=4)
        if text:
            mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2 + 0.22
            ax.text(mid_x, mid_y, text, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=color, lw=1),
                    zorder=5)

    # Helper to draw pseudo 3D cuboid for feature maps
    def draw_cuboid(cx, cy, w, h, d, face_color="#6366f1", edge_color="#4338ca", alpha=0.85):
        front = patches.Rectangle((cx - w/2, cy - h/2), w, h,
                                  fc=face_color, ec=edge_color, lw=1.2, alpha=alpha, zorder=6)
        ax.add_patch(front)
        top = Polygon([
            [cx - w/2, cy + h/2],
            [cx - w/2 + d*0.6, cy + h/2 + d*0.5],
            [cx + w/2 + d*0.6, cy + h/2 + d*0.5],
            [cx + w/2, cy + h/2]
        ], closed=True, fc=face_color, ec=edge_color, lw=1.2, alpha=alpha*0.85, zorder=6)
        ax.add_patch(top)
        side = Polygon([
            [cx + w/2, cy - h/2],
            [cx + w/2, cy + h/2],
            [cx + w/2 + d*0.6, cy + h/2 + d*0.5],
            [cx + w/2 + d*0.6, cy - h/2 + d*0.5]
        ], closed=True, fc=face_color, ec=edge_color, lw=1.2, alpha=alpha*0.7, zorder=6)
        ax.add_patch(side)

    # -------------------------------------------------------------
    # STAGE 1: INPUT STREAM (LEFT, x = 0.5 to 4.1)
    # -------------------------------------------------------------
    draw_card(0.5, 0.6, 3.6, 9.8, "STAGE 1: INPUT STREAM", "Differential Micro-Motion Extraction", "#0284c7", "#ffffff", lw=2.2)

    # Sub-block 1: Video CASME II (y = 6.8 to 9.2, h = 2.4)
    draw_card(0.75, 6.8, 3.1, 2.4, "Raw CASME II Video Clip", "Onset to Offset (L Frames)", "#0369a1", "#f0f9ff")
    ax.text(2.3, 7.85, "Uniform Temporal Sampling\nT = 16 Distributed Frames\nFace Crop Resolution: 128 × 128 px",
            ha='center', va='center', fontsize=8.8, color="#0c4a6e")
    # Mini frame representation
    for i in range(5):
        rect = patches.Rectangle((1.4 + i*0.36, 7.05), 0.28, 0.38,
                                 fc="#bae6fd", ec="#0284c7", lw=1, zorder=6)
        ax.add_patch(rect)
    ax.text(2.3, 6.95, "16 Temporal Slices", ha='center', va='center', fontsize=7.2, color="#0369a1")

    draw_arrow(2.3, 6.8, 2.3, 6.05, "#0284c7")

    # Sub-block 2: TV-L1 Optical Flow (y = 3.8 to 6.05, h = 2.25)
    draw_card(0.75, 3.8, 3.1, 2.25, "TV-L1 Dual Optical Flow", "Relative Muscle Motion Field", "#0369a1", "#f0f9ff")
    ax.text(2.3, 4.85, "Reference: Onset Frame (t = 0)\nVectors: u_x, u_y & Magnitude\nEliminates Subject Static Identity Bias",
            ha='center', va='center', fontsize=8.2, color="#0c4a6e")
    ax.text(2.3, 4.15, "Mag = sqrt(u_x² + u_y²)", ha='center', va='center',
            fontsize=8, fontweight='bold', color="#0369a1",
            bbox=dict(boxstyle="round,pad=0.15", fc="#e0f2fe", ec="#0284c7", lw=0.8))

    draw_arrow(2.3, 3.8, 2.3, 3.05, "#0284c7")

    # Sub-block 3: Motion Volume Tensor (y = 0.85 to 3.05, h = 2.2)
    draw_card(0.75, 0.85, 3.1, 2.2, "Optical Flow Motion Tensor", "Input to 3D CNN Backbone", "#1d4ed8", "#eff6ff")
    ax.text(2.3, 2.0, "Input Tensor Dimensions:\n[Batch, 3, 16, 128, 128]",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color="#1e40af")
    ax.text(2.3, 1.35, "Channels: (u, v, mag)\nTemporal: T = 16 | Spatial: 128×128",
            ha='center', va='center', fontsize=8, color="#3b82f6")

    # Direct connector from Stage 1 to Stage 2 Conv3D Stem
    draw_arrow(3.85, 1.95, 4.85, 1.95, "#0284c7", lw=2.4, text="[3, 16, 128, 128]")

    # -------------------------------------------------------------
    # STAGE 2: 3D CNN BACKBONE (MIDDLE, x = 4.6 to 14.9)
    # -------------------------------------------------------------
    draw_card(4.6, 0.6, 10.3, 9.8, "STAGE 2: SPATIOTEMPORAL 3D CNN BACKBONE (R3D-18)", "Hierarchical Spatiotemporal Motion Learning", "#4f46e5", "#ffffff", lw=2.2)

    # Column 1: Conv3D Stem (y = 0.85 to 9.2, h = 8.35)
    draw_card(4.85, 0.85, 2.25, 8.35, "Conv3D Stem", "Initial Feature Extractor", "#4338ca", "#f5f3ff")
    ax.text(5.97, 8.05, "Stem 3D Convolution\nKernel: 3 × 7 × 7\nStride: (1, 2, 2)\nPadding: (1, 3, 3)",
            ha='center', va='center', fontsize=8.2, color="#312e81")
    ax.text(5.97, 6.75, "BatchNorm3d\n+\nReLU Activation",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color="#4f46e5",
            bbox=dict(boxstyle="round,pad=0.2", fc="#ede9fe", ec="#8b5cf6", lw=0.8))
    # 3D Cuboid illustration
    draw_cuboid(5.97, 5.15, 0.8, 1.0, 0.4, face_color="#c7d2fe", edge_color="#4f46e5")
    ax.text(5.97, 3.75, "Spatial Downsampling:\n128×128  ➔  64×64\nTemporal Preserved (T=16)",
            ha='center', va='center', fontsize=7.8, color="#475569")
    ax.text(5.97, 1.75, "Feature Map:\n[64, 16, 64, 64]",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color="#4338ca",
            bbox=dict(boxstyle="round,pad=0.2", fc="#e0e7ff", ec="#6366f1", lw=1))

    draw_arrow(7.1, 5.15, 7.45, 5.15, "#4f46e5", lw=2)

    # Column 2: ResNet Layer 1 & 2
    draw_card(7.45, 0.85, 2.3, 8.35, "ResNet Layer 1 & 2", "Spatiotemporal Residuals", "#4338ca", "#f5f3ff")
    ax.text(8.6, 8.05, "Layer 1 (2x BasicBlock3D)\nChannels: 64 ➔ 64\nOutput: [64, 16, 64, 64]",
            ha='center', va='center', fontsize=8, color="#312e81")
    ax.text(8.6, 6.8, "Residual Shortcut:\nx + F(x)\nPreserves Subtle Motion",
            ha='center', va='center', fontsize=7.8, color="#6366f1",
            bbox=dict(boxstyle="round,pad=0.2", fc="#ede9fe", ec="#a78bfa", lw=0.8))
    # 3D Cuboid illustration
    draw_cuboid(8.6, 5.15, 0.7, 0.8, 0.4, face_color="#a5b4fc", edge_color="#4338ca")
    ax.text(8.6, 3.75, "Layer 2 (2x BasicBlock3D)\nChannels: 64 ➔ 128\nStride: (1, 2, 2)\nSpatial: 64×64 ➔ 32×32",
            ha='center', va='center', fontsize=8, color="#312e81")
    ax.text(8.6, 1.75, "Feature Map:\n[128, 16, 32, 32]",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color="#4338ca",
            bbox=dict(boxstyle="round,pad=0.2", fc="#e0e7ff", ec="#6366f1", lw=1))

    draw_arrow(9.75, 5.15, 10.1, 5.15, "#4f46e5", lw=2)

    # Column 3: ResNet Layer 3 & 4
    draw_card(10.1, 0.85, 2.3, 8.35, "ResNet Layer 3 & 4", "Deep Semantics & AU Cues", "#4338ca", "#f5f3ff")
    ax.text(11.25, 8.05, "Layer 3 (2x BasicBlock3D)\nChannels: 128 ➔ 256\nStride: (1, 2, 2)\nOutput: [256, 16, 16, 16]",
            ha='center', va='center', fontsize=8, color="#312e81")
    ax.text(11.25, 6.8, "Expanded Receptive Field:\nCaptures Coordinated\nFacial Muscle Dynamics",
            ha='center', va='center', fontsize=7.8, color="#6366f1",
            bbox=dict(boxstyle="round,pad=0.2", fc="#ede9fe", ec="#a78bfa", lw=0.8))
    # 3D Cuboid illustration
    draw_cuboid(11.25, 5.15, 0.55, 0.6, 0.35, face_color="#818cf8", edge_color="#3730a3")
    ax.text(11.25, 3.75, "Layer 4 (2x BasicBlock3D)\nChannels: 256 ➔ 512\nStride: (1, 2, 2)\nSpatial: 16×16 ➔ 8×8",
            ha='center', va='center', fontsize=8, color="#312e81")
    ax.text(11.25, 1.75, "Feature Map:\n[512, 16, 8, 8]",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color="#4338ca",
            bbox=dict(boxstyle="round,pad=0.2", fc="#e0e7ff", ec="#6366f1", lw=1))

    draw_arrow(12.4, 5.15, 12.75, 5.15, "#4f46e5", lw=2)

    # Column 4: Pooling & Regularization
    draw_card(12.75, 0.85, 1.85, 8.35, "Pooling & Dropout", "Feature Aggregation", "#7c3aed", "#faf5ff")
    ax.text(13.67, 8.05, "AdaptiveAvgPool3d\nOutput: (1, 1, 1)\n\nGlobal Spatiotemporal\nFeature Compression\n[512, 16, 8, 8] ➔ [512]",
            ha='center', va='center', fontsize=7.8, color="#5b21b6")
    ax.text(13.67, 5.5, "Flattening ➔ 512-D",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color="#7c3aed",
            bbox=dict(boxstyle="round,pad=0.2", fc="#f3e8ff", ec="#c084fc", lw=0.8))
    ax.text(13.67, 4.0, "Shared Dropout\np = 0.5 Probability\nPrevents Co-adaptation\nof Shared Features",
            ha='center', va='center', fontsize=7.8, color="#5b21b6")
    ax.text(13.67, 1.75, "Latent Vector:\n[Batch, 512]",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color="#7c3aed",
            bbox=dict(boxstyle="round,pad=0.2", fc="#e9d5ff", ec="#a855f7", lw=1))

    # Latent Vector Bridge between Stage 2 and Stage 3
    draw_arrow(14.6, 5.15, 15.0, 5.15, "#4f46e5", lw=2.5)
    ax.text(15.35, 5.15, "512-D Latent\nFeature", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color="#4f46e5",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#4f46e5", lw=1.2), zorder=6)

    # Branching arrows from Latent badge to Stage 3 Dual Heads
    draw_arrow(15.7, 5.45, 16.2, 7.3, "#059669", lw=2.2, rad=-0.15)
    draw_arrow(15.7, 4.85, 16.2, 3.6, "#d97706", lw=2.2, rad=0.15)

    # -------------------------------------------------------------
    # STAGE 3: MULTI-TASK DUAL HEADS (RIGHT, x = 16.0 to 22.9)
    # -------------------------------------------------------------
    draw_card(16.0, 0.6, 6.9, 9.8, "STAGE 3: MULTI-TASK DUAL HEADS", "Joint Emotion Classification & Facial Action Unit Supervision", "#0f172a", "#ffffff", lw=2.2)

    # Head 1: Primary Emotion Head (Top, y = 5.4 to 9.2, h = 3.8)
    draw_card(16.3, 5.4, 6.3, 3.8, "Primary Emotion Classification Head", "Official MEGC Benchmark Objective (5 Classes)", "#059669", "#ecfdf5")
    ax.text(19.45, 8.0, "Linear(512 ➔ 5)  +  Softmax Activation", ha='center', va='center',
            fontsize=10.2, fontweight='bold', color="#047857")
    
    # 5 Emotion Class Badges
    emotions = [("Happiness", "#10b981"), ("Disgust", "#ef4444"), ("Repression", "#8b5cf6"),
                ("Surprise", "#f59e0b"), ("Others", "#64748b")]
    for idx, (emo, clr) in enumerate(emotions):
        bx = 16.85 + idx * 1.05
        by = 7.3
        ax.text(bx, by, emo, ha='center', va='center', fontsize=7.5, fontweight='bold', color=clr,
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=clr, lw=1), zorder=5)

    ax.text(19.45, 6.35, "Loss: L_CE (Inverse Class Frequency Weights + Label Smoothing 0.1)\nTarget Metrics: Macro-F1 (UF1: 0.7211) & Recall (UAR: 0.7378)",
            ha='center', va='center', fontsize=8.2, color="#065f46",
            bbox=dict(boxstyle="round,pad=0.25", fc="#d1fae5", ec="#34d399", lw=0.8))

    # Head 2: Auxiliary Action Unit Head (Bottom, y = 2.1 to 5.15, h = 3.05)
    draw_card(16.3, 2.1, 6.3, 3.05, "Auxiliary Facial Action Unit (AU) Head", "Anatomical Muscle Regularization (11 FACS AUs)", "#d97706", "#fffbeb")
    ax.text(19.45, 4.25, "Linear(512 ➔ 11)  +  Sigmoid (Multi-Label)", ha='center', va='center',
            fontsize=10.2, fontweight='bold', color="#b45309")
    
    # 11 AU Chips in 2 rows
    row1_aus = ["AU1", "AU2", "AU4", "AU5", "AU7", "AU9"]
    for idx, au in enumerate(row1_aus):
        bx = 16.9 + idx * 0.85
        by = 3.65
        ax.text(bx, by, au, ha='center', va='center', fontsize=7.2, fontweight='bold', color="#b45309",
                bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#f59e0b", lw=0.8), zorder=5)
    row2_aus = ["AU10", "AU12", "AU14", "AU15", "AU17"]
    for idx, au in enumerate(row2_aus):
        bx = 17.3 + idx * 0.88
        by = 3.2
        ax.text(bx, by, au, ha='center', va='center', fontsize=7.2, fontweight='bold', color="#b45309",
                bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#f59e0b", lw=0.8), zorder=5)

    ax.text(19.45, 2.5, "Loss: L_BCE (Positive-Weight Balanced for Sparse AUs)\nAuxiliary Regularization Weight: lambda_AU = 0.5 (Prevents Identity Overfitting)",
            ha='center', va='center', fontsize=8.2, color="#92400e",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fef3c7", ec="#f59e0b", lw=0.8))

    # Joint Objective Card at Bottom of Stage 3 (y = 0.85 to 1.85, h = 1.0)
    draw_card(16.3, 0.85, 6.3, 1.0, "", "", "#dc2626", "#fee2e2", rad=0.1, lw=1.5)
    ax.text(19.45, 1.45, "JOINT MULTI-TASK OBJECTIVE FUNCTION:", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color="#991b1b")
    ax.text(19.45, 1.1, "L_total = L_CE(Emotion) + 0.5 × L_BCE(Action Units)", ha='center', va='center',
            fontsize=11.2, fontweight='bold', color="#b91c1c")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Successfully saved final architecture diagram to {output_path}")

if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    os.makedirs("runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5", exist_ok=True)
    create_architecture_diagram("docs/architecture_r3d_multitask_au.png")
    create_architecture_diagram("runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/architecture.png")
