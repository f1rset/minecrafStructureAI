import os
import shutil
import subprocess
import fal_client
import trimesh
import requests
from pathlib import Path
from flask import Flask, render_template, request, send_file, flash, redirect

app = Flask(__name__)
app.secret_key = "super_secret_key"

# Configuration
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure FAL_KEY is set
if not os.environ.get("FAL_KEY"):
    print("WARNING: FAL_KEY environment variable is not set.")


def stl_to_schematic(stl_path: Path, out_dir: Path, resolution=128):
    """
    User-provided function to convert STL to Minecraft Schematic using binvox.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve absolute paths
    # Assumes 'binvox' executable is in the same directory as this script
    binvox_path = Path(__file__).parent / "binvox"
    stl_absolute = stl_path.resolve()

    if not stl_absolute.exists():
        raise FileNotFoundError(f"STL file missing at: {stl_absolute}")

    # Check if binvox executable exists
    if not binvox_path.exists():
        raise FileNotFoundError(
            f"Binvox executable not found at: {binvox_path}. Please download it."
        )

    print(f"[+] Running binvox on {stl_absolute}")

    # Note: binvox outputs to the same directory as the input file
    try:
        subprocess.run(
            [
                str(binvox_path),
                "-d",
                str(resolution),
                "-t",
                "schematic",  # This generates a legacy .schematic file
                str(stl_absolute),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print(f"Binvox Error: {e.stderr.decode()}")
        raise RuntimeError("Failed to run binvox.")

    # 2. Handle the output file
    # binvox appends _1.schematic (or similar) or just replaces extension depending on version/flags
    # Usually it creates [filename].schematic inside the same folder as input
    generated_schem = stl_absolute.with_suffix(".schematic")

    if generated_schem.exists():
        destination = out_dir / generated_schem.name
        shutil.move(str(generated_schem), str(destination))
        print(f"[✓] Schematic moved to {destination}")
        return destination
    else:
        print("[!] Warning: binvox finished but no schematic file was found.")
        return None


def glb_to_stl(glb_path: Path) -> Path:
    """Converts a GLB file to STL using Trimesh."""
    print(f"[+] Converting GLB to STL: {glb_path}")
    mesh = trimesh.load(glb_path)

    # If the scene has multiple geometries, combine them
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(trimesh.util.concatenate(g) for g in mesh.geometry.values())
        )

    stl_path = glb_path.with_suffix(".stl")
    mesh.export(stl_path)
    return stl_path


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file:
            # 1. Save User Image
            img_path = app.config["UPLOAD_FOLDER"] / file.filename
            file.save(img_path)

            try:
                # 2. Upload to fal.ai and Generate 3D Model
                print("[+] Uploading image to fal.ai...")
                url = fal_client.upload_file(img_path)

                print("[+] Triggering Hunyuan3D v2...")
                handler = fal_client.submit(
                    "fal-ai/hunyuan3d/v2", arguments={"input_image_url": url}
                )

                result = handler.get()

                # Extract GLB URL (The API returns 'model_mesh' -> 'url')
                glb_url = result["model_mesh"]["url"]

                # 3. Download the GLB
                import requests

                glb_response = requests.get(glb_url)
                glb_filename = f"{img_path.stem}.glb"
                glb_path = app.config["UPLOAD_FOLDER"] / glb_filename

                with open(glb_path, "wb") as f:
                    f.write(glb_response.content)

                # 4. Convert GLB -> STL
                stl_path = glb_to_stl(glb_path)

                # 5. Convert STL -> Minecraft Schematic (using your function)
                schematic_path = stl_to_schematic(stl_path, app.config["UPLOAD_FOLDER"])

                if schematic_path and schematic_path.exists():
                    # Helper: Rename to .structure if user prefers (though format is technically schematic)
                    # Note: Genuine .structure files are NBT. binvox produces legacy .schematic (WorldEdit).
                    # We serve it as .schematic to ensure it works with tools like WorldEdit/MCEdit.
                    return send_file(
                        schematic_path,
                        as_attachment=True,
                        download_name=f"{img_path.stem}.schematic",
                    )
                else:
                    flash("Failed to generate schematic file.")

            except Exception as e:
                flash(f"Error: {str(e)}")
                print(e)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
