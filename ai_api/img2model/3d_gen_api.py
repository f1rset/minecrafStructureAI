#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

import torch
import trimesh
import numpy as np

# DUSt3R imports
sys.path.append("dust3r")
from dust3r.demo import load_images
from dust3r.inference import inference  # <--- Import load_model here
from dust3r.image_pairs import make_pairs
from dust3r.model import AsymmetricCroCo3DStereo
import shutil

# ---------------- CONFIG ----------------

# Assuming this checkpoint is actually a ViT-Base model based on the warnings
DUST3R_CKPT = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- DUST3R → STL ----------------


def image_to_stl(image_paths: list[Path], out_stl: Path):
    if len(image_paths) < 2:
        print("[!] Error: DUSt3R requires at least 2 images to generate 3D geometry.")
        return False

    # ... (Model loading code remains exactly the same as your last working run) ...
    # ... (Keep the "RoPE100" and "model.enc_pos_embed = None" fix) ...

    # [PASTE YOUR WORKING MODEL LOADING CODE HERE]
    print("[+] Loading DUSt3R model")
    model = AsymmetricCroCo3DStereo(
        img_size=(512, 512),
        patch_size=16,
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_depth=12,
        dec_embed_dim=768,
        dec_num_heads=12,
        output_mode="pts3d",
        head_type="dpt",
        pos_embed="RoPE100",
    )
    model.enc_pos_embed = None
    ckpt = torch.load(DUST3R_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(DEVICE)
    model.eval()

    print(f"[+] Loading {len(image_paths)} images")
    images = load_images([str(p) for p in image_paths], size=512)

    print("[+] Creating image pairs")
    pairs = make_pairs(images, scene_graph="complete", symmetrize=True)

    print(f"[+] Running DUSt3R inference on {len(pairs)} pairs")
    with torch.no_grad():
        result = inference(pairs, model, DEVICE)

    # ---------------------------------------------------------
    # DEBUG & EXTRACTION LOGIC
    # ---------------------------------------------------------
    print("\n[DEBUG] Inference result keys:", result.keys())

    # Try to find the point clouds in the dictionary
    all_pts = []
    all_cols = []

    # CASE A: Standard Dictionary Output (common in some versions)
    # It might contain a list of dictionaries under 'preds' or similar
    if "pts3d" in result:
        # If it returns a single batch result directly
        print("[+] Found 'pts3d' key directly")
        pts_batch = result["pts3d"]  # Shape likely (B, H, W, 3)
        colors_batch = result.get("colors", images[0][None, ...])  # Fallback for colors

        for i in range(len(pts_batch)):
            all_pts.append(pts_batch[i].detach().cpu().numpy().reshape(-1, 3))
            # Handle colors (might need broadcasting or be in the dict)
            # For now, let's just use the input images as colors if missing
            if isinstance(colors_batch, list):
                all_cols.append(colors_batch[i].reshape(-1, 3))
            elif isinstance(colors_batch, torch.Tensor):
                all_cols.append(colors_batch[i].detach().cpu().numpy().reshape(-1, 3))
            else:
                all_cols.append(images[i].reshape(-1, 3))

    # CASE B: It is a dictionary of pairs -> predictions
    # e.g. {(0, 1): {'pts3d': ...}, (1, 0): ...}
    elif isinstance(list(result.keys())[0], tuple):
        print("[+] Found pair-based dictionary")
        for (view1, view2), output_data in result.items():
            # output_data is likely the prediction dict for this pair
            if "pts3d" in output_data:
                p = output_data["pts3d"].detach().cpu().numpy().reshape(-1, 3)
                all_pts.append(p)
                # We reuse the original images for colors to be safe
                # view1 is the index of the first image in the pair
                c = images[view1].reshape(-1, 3)
                all_cols.append(c)

    # CASE C: It acts like a list (but is a dict?) - unlikely but possible
    else:
        print("[!] Unknown output structure. Trying generic fallback.")
        # Attempt to inspect values
        first_val = list(result.values())[0]
        print(f"[DEBUG] First value type: {type(first_val)}")
        if hasattr(first_val, "keys"):
            print(f"[DEBUG] First value keys: {first_val.keys()}")

        return False  # Stop here so you can see the debug print

    if not all_pts:
        print("[!] Could not extract points. See debug info above.")
        return False

    # Fusion
    pts = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)

    print(f"[+] Final Point Cloud Size: {pts.shape[0]} points")
    print("[+] Converting point cloud → mesh")

    cloud = trimesh.points.PointCloud(pts, colors=colors)
    mesh = cloud.convex_hull
    mesh.export(out_stl)

    print(f"[✓] STL saved: {out_stl}")
    return True


# ---------------- BINVOX → SCHEMATIC ----------------


def stl_to_schematic(stl_path: Path, out_dir: Path, resolution=128):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve absolute paths
    binvox_path = Path(__file__).parent / "binvox"
    stl_absolute = stl_path.resolve()

    if not stl_absolute.exists():
        raise FileNotFoundError(f"STL file missing at: {stl_absolute}")

    print(f"[+] Running binvox on {stl_absolute}")

    subprocess.run(
        [
            str(binvox_path),
            "-d",
            str(resolution),
            "-t",
            "schematic",
            str(stl_absolute),  # <--- CHANGE: Use full path, not stl_path.name
        ],
        check=True,
    )

    # 2. Handle the output file
    # binvox always saves the output in the SAME directory as the source STL
    # We need to find it there and move it to your 'out_dir'
    generated_schem = stl_absolute.with_suffix(".schematic")

    if generated_schem.exists():
        destination = out_dir / generated_schem.name
        shutil.move(str(generated_schem), str(destination))
        print(f"[✓] Schematic moved to {destination}")
    else:
        print("[!] Warning: binvox finished but no schematic file was found.")


# ---------------- MAIN ----------------


def main():
    parser = argparse.ArgumentParser(
        description="Multi-View Images → STL → Minecraft schematic"
    )

    # CHANGE: nargs='+' allows accepting multiple files (e.g., img1.png img2.png)
    parser.add_argument(
        "images", type=Path, nargs="+", help="List of image paths (at least 2 required)"
    )
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--res", type=int, default=128)

    args = parser.parse_args()

    args.out.mkdir(exist_ok=True)
    stl_path = args.out / "mesh.stl"

    # Pass the list of images
    success = image_to_stl(args.images, stl_path)

    # Only run binvox if the STL was actually generated
    if success:
        stl_to_schematic(stl_path, args.out, args.res)
    else:
        print("[-] Pipeline aborted due to previous errors.")


if __name__ == "__main__":
    main()
