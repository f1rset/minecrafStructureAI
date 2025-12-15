import os
import subprocess
import requests
import torch
from pathlib import Path
from flask import (
    Flask,
    render_template_string,
    request,
    send_from_directory,
    flash,
    redirect,
    url_for,
)

import fal_client
import trimesh
from diffusers import StableDiffusionPipeline
from rembg import remove

app = Flask(__name__)
app.secret_key = "super_secret_key"

# --- Configuration ---
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.environ.get("FAL_KEY"):
    print("WARNING: FAL_KEY environment variable is not set.")

print("[+] Loading Stable Diffusion Pipeline...")
MODEL_ID = "Lykon/dreamshaper-8"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    )
    if device == "cuda":
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    print("[+] Model loaded successfully.")
except Exception as e:
    print(f"[!] Failed to load Stable Diffusion: {e}")
    pipe = None

def generate_asset_from_text(prompt: str, filename: str) -> Path:
    """Generates an image from text using SD + Rembg."""
    if pipe is None:
        raise RuntimeError("Stable Diffusion pipeline not loaded.")

    positive_prompt = f"isometric 3d asset, {prompt}, plain white background, minimal style, single object, unreal engine render, 4k"
    negative_prompt = "shadows, complex background, noise, messy, text, watermark, floor, ground, wall, realistic photo"

    print(f"[+] Generating image for: {prompt}")
    result = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
    ).images[0]

    print("[+] Removing background...")
    clean_image = remove(result)

    out_path = app.config["UPLOAD_FOLDER"] / filename
    clean_image.save(out_path)
    return out_path


def glb_to_stl(glb_path: Path) -> Path:
    """Converts a GLB file to STL using Trimesh."""
    print(f"[+] Converting GLB to STL: {glb_path}")
    mesh = trimesh.load(glb_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(trimesh.util.concatenate(g) for g in mesh.geometry.values())
        )
    stl_path = glb_path.with_suffix(".stl")
    mesh.export(stl_path)
    return stl_path


def stl_to_struct(stl_path: Path, out_dir: Path, resolution=128):
    binvox_path = Path(__file__).parent / "binvox"
    if not binvox_path.exists():
        binvox_path = Path(__file__).parent / "binvox.exe"

    if not binvox_path.exists():
        raise FileNotFoundError("binvox executable not found.")

    print(f"[+] Running binvox on {stl_path}")
    subprocess.run(
        [str(binvox_path), "-d", str(resolution), "-t", "schematic", str(stl_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    generated_schem = stl_path.with_suffix(".schematic")

    if generated_schem.exists():
        return generated_schem
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_path = None

        if "prompt" in request.form and request.form["prompt"].strip():
            prompt_text = request.form["prompt"]
            safe_name = "".join(x for x in prompt_text if x.isalnum())[:10]
            filename = f"{safe_name}.png"
            try:
                img_path = generate_asset_from_text(prompt_text, filename)
            except Exception as e:
                flash(f"Generation Error: {e}")
                return redirect(request.url)

        elif "image" in request.files:
            file = request.files["image"]
            if file.filename:
                img_path = app.config["UPLOAD_FOLDER"] / file.filename
                file.save(img_path)

        if not img_path:
            flash("No image or prompt provided.")
            return redirect(request.url)

        try:
            # 1. Upload to Fal.ai (Hunyuan3D)
            print("[+] Uploading to Fal.ai...")
            url = fal_client.upload_file(img_path)

            print("[+] Generating 3D Model...")
            handler = fal_client.submit(
                "fal-ai/hunyuan3d/v2", arguments={"input_image_url": url}
            )
            result = handler.get()
            glb_url = result["model_mesh"]["url"]

            # 2. Download GLB
            glb_response = requests.get(glb_url)
            glb_path = img_path.with_suffix(".glb")
            with open(glb_path, "wb") as f:
                f.write(glb_response.content)

            # 3. GLB -> STL -> STRUCT
            stl_path = glb_to_stl(glb_path)
            struct_path = stl_to_struct(stl_path, app.config["UPLOAD_FOLDER"])

            if struct_path:
                # Redirect to preview page
                return render_template_string(
                    PREVIEW_TEMPLATE,
                    image_file=img_path.name,
                    struct_file=struct_path.name,
                )
            else:
                flash("Failed to generate structure file.")

        except Exception as e:
            flash(f"Processing Error: {str(e)}")
            print(e)

    return render_template_string(INDEX_TEMPLATE)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serves files from the upload directory so we can preview them."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


INDEX_TEMPLATE = """
<!doctype html>
<title>AI to Minecraft Structure</title>
<style>
    body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }
    input[type=text], input[type=file] { width: 100%; padding: 10px; margin: 10px 0; }
    button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
    .divider { text-align: center; margin: 20px 0; color: #666; }
</style>
<h1>Generate 3D .struct</h1>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <div style="background: #f8d7da; padding: 10px; margin-bottom: 20px;">
    {% for message in messages %}{{ message }}<br>{% endfor %}
    </div>
  {% endif %}
{% endwith %}

<form method=post enctype=multipart/form-data>
  <h3>Option 1: Type a prompt</h3>
  <input type="text" name="prompt" placeholder="e.g. A stone castle tower">
  
  <div class="divider">- OR -</div>
  
  <h3>Option 2: Upload an image</h3>
  <input type="file" name="image">
  
  <br><br>
  <button type="submit">Generate 3D Model</button>
</form>
"""

PREVIEW_TEMPLATE = """
<!doctype html>
<title>Preview Result</title>
<style>
    body { font-family: sans-serif; text-align: center; padding: 50px; }
    img { max-width: 400px; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .btn { display: inline-block; margin-top: 20px; padding: 15px 30px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }
    .back { display: block; margin-top: 20px; color: #666; }
</style>
<h1>Result Ready!</h1>
<p>Here is the generated concept image:</p>
<img src="{{ url_for('uploaded_file', filename=image_file) }}" alt="Generated Image">
<br>
<a href="{{ url_for('uploaded_file', filename=struct_file) }}" class="btn">Download .struct file</a>
<a href="/" class="back">← Create Another</a>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)

