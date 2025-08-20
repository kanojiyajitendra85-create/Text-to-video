import os
import tempfile
import shutil
import streamlit as st
from typing import List, Optional
from pathlib import Path
import subprocess
import time

# Media libs
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Optional heavy imports will be imported lazily when needed

# ---------- Configuration / Helpers ----------
DEFAULT_FPS = 12
FRAME_SIZE = (640, 384)  # width, height

st.set_page_config(page_title="Text→Video→Anime", layout="wide")
st.title("🎬 Text → Video → Anime (streamlined)")
st.write("This app orchestrates text->image/video (diffusers or external API), stitches frames into a single MP4, adds audio, and applies an anime/cartoon style.")

# Sidebar options
st.sidebar.header("Settings")
use_local_diffusers = st.sidebar.checkbox("Use local Stable Diffusion (requires GPU)", value=False)
external_provider = st.sidebar.selectbox("External provider (if not local)", ["replicate", "stability", "none"])
replicate_token = st.sidebar.text_input("Replicate API Token (optional)", type="password")
stability_key = st.sidebar.text_input("Stability API Key (optional)", type="password")

apply_animegan = st.sidebar.checkbox("Apply AnimeGAN (ONNX) if available", value=True)
onnx_path = st.sidebar.text_input("AnimeGAN ONNX path (optional)")
use_opencv_fallback = st.sidebar.checkbox("Use OpenCV cartoonizer fallback", value=True)

scene_seconds = st.sidebar.slider("Seconds per scene (target)", 2, 8, 4)
fps = st.sidebar.slider("FPS for final video", 6, 30, DEFAULT_FPS)

# Audio options
st.sidebar.header("Audio")
use_tts = st.sidebar.checkbox("Generate TTS audio from script (Google TTS/gTTS)", value=False)
audio_upload = st.sidebar.file_uploader("Upload audio file (optional)", type=["mp3","wav","m4a"])

# Fonts for PIL text drawing
try:
    default_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
except Exception:
    default_font = ImageFont.load_default()

# ---------- Core Processing Functions ----------

def ensure_ffmpeg_available():
    try:
        subprocess.check_output(["ffmpeg", "-version"])  # type: ignore
    except Exception as e:
        st.error("ffmpeg is required and not found on PATH. Install ffmpeg before running this app.")
        raise


# Simple OpenCV cartoonizer for fallback
def cartoonize_image_cv(img_bgr: np.ndarray,
                        edge_ksize: int = 5,
                        bilateral_iterations: int = 4,
                        bilateral_sigma: int = 75,
                        downscale: int = 2) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (w // downscale, h // downscale))
    filtered = small
    for _ in range(bilateral_iterations):
        filtered = cv2.bilateralFilter(filtered, d=9, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, edge_ksize)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9, C=2)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(filtered, edges_color)
    cartoon_full = cv2.resize(cartoon, (w, h), interpolation=cv2.INTER_LINEAR)
    return cartoon_full


# AnimeGAN ONNX inference (optional)
def apply_animegan_onnx_frame(frame_bgr: np.ndarray, session) -> np.ndarray:
    import numpy as np
    inp_name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape  # e.g., [1,3,512,512]
    _, c, ih, iw = shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih)).astype(np.float32) / 127.5 - 1.0
    nchw = np.transpose(resized, (2,0,1))[None, ...]
    out = session.run(None, {inp_name: nchw})[0]
    out = out[0]
    out = np.transpose(out, (1,2,0))
    out = ((out + 1.0) * 127.5).clip(0,255).astype('uint8')
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    out_bgr = cv2.resize(out_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return out_bgr


# Generate frames using local diffusers (simple img2img-style per scene)
# This is a best-effort example; running requires installation of diffusers, accelerate, and models.

def generate_frames_local_sd(prompt: str, frames_per_scene: int, seed: int = 0) -> List[Path]:
    """
    Generate frames_per_scene images for the given prompt using a Stable Diffusion image model.
    This function expects diffusers and a suitable model to be installed. This routine keeps
    a small motion by doing small img2img steps or strength jittering to produce temporal variation.
    """
    try:
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
        import torch
    except Exception as e:
        st.error("diffusers and torch are required for local generation. Install them and provide GPU.")
        raise

    # NOTE: pick a model that you have access to, e.g., 'runwayml/stable-diffusion-v1-5' or an anime-specific SDXL
    model_id = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=hf_token)
    pipe = pipe.to(device)

    out_paths = []
    tmpdir = Path(tempfile.mkdtemp())
    for i in range(frames_per_scene):
        this_seed = seed + i
        generator = torch.Generator(device).manual_seed(this_seed)
        img = pipe(prompt, guidance_scale=7.5, num_inference_steps=20, generator=generator).images[0]
        # resize to FRAME_SIZE
        img = img.resize(FRAME_SIZE)
        p = tmpdir / f"frame_{i:04d}.png"
        img.save(p)
        out_paths.append(p)
    return out_paths


# External provider placeholder

def generate_video_external(prompt: str, out_path: str, provider: str = "replicate", token: Optional[str] = None):
    """
    Calls an external provider API to generate a short MP4 clip for the prompt.
    Implementations for replicate/stability require reading their docs and filling request params.
    This placeholder raises until implemented by the user.
    """
    raise NotImplementedError("External provider integration must be implemented for your chosen provider.")


# Stitch frames into an MP4 with optional audio

def frames_to_video(frame_paths: List[Path], out_path: str, fps: int = DEFAULT_FPS, audio_path: Optional[str] = None):
    imgs = [str(p) for p in frame_paths]
    clip = ImageSequenceClip(imgs, fps=fps)
    if audio_path:
        clip = clip.set_audio(AudioFileClip(audio_path))
    clip.write_videofile(out_path, codec="libx264", audio_codec='aac' if audio_path else None)
    return out_path


# High-level pipeline orchestration

def process_script_to_video(script_text: str,
                            work_dir: str,
                            per_scene_seconds: int = 4,
                            fps: int = DEFAULT_FPS,
                            local_sd: bool = False,
                            external_provider_name: str = "replicate",
                            provider_token: Optional[str] = None,
                            apply_anime: bool = True,
                            onnx_model_path: Optional[str] = None,
                            tts_audio_path: Optional[str] = None,
                            user_audio_path: Optional[str] = None):
    """
    Orchestrate scene splitting, frame generation, assembly, audio, and stylization.
    Returns paths: (raw_video, stylized_video)
    """
    ensure_ffmpeg_available()

    tmp = Path(work_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    # split scenes by lines that start with 'Scene' or blank-line segmentation
    lines = [ln.strip() for ln in script_text.split('\n') if ln.strip()]
    scenes = []
    cur = []
    for ln in lines:
        if ln.lower().startswith("scene") and cur:
            scenes.append(" ".join(cur))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        scenes.append(" ".join(cur))

    scene_clips = []
    raw_scene_paths = []

    # For ONNX anime model loading
    onnx_session = None
    if apply_anime and onnx_model_path:
        try:
            import onnxruntime as ort
            onnx_session = ort.InferenceSession(onnx_model_path)
        except Exception as e:
            st.warning(f"Failed to load ONNX model: {e}. Will fallback to OpenCV cartoonizer.")
            onnx_session = None

    for idx, scene in enumerate(scenes):
        st.info(f"Generating scene {idx+1}/{len(scenes)}: {scene[:80]}")
        # Generate frames for this scene
        frames_needed = max(1, int(per_scene_seconds * fps))
        frame_paths = []

        if local_sd:
            # Local frame generation (slow, GPU)
            frame_paths = generate_frames_local_sd(scene, frames_per_scene=frames_needed, seed=idx*100)
        else:
            # Use external provider to get an MP4 clip per scene, then extract frames
            tmp_scene_video = tmp / f"scene_{idx+1}.mp4"
            generate_video_external(scene, str(tmp_scene_video), provider=external_provider_name, token=provider_token)
            # extract frames with ffmpeg
            frame_dir = tmp / f"frames_{idx+1}"
            frame_dir.mkdir(exist_ok=True)
            cmd = [
                "ffmpeg", "-y", "-i", str(tmp_scene_video),
                "-vf", f"fps={fps}",
                str(frame_dir / "frame_%06d.png")
            ]
            subprocess.check_call(cmd)
            frame_paths = sorted(frame_dir.glob("*.png"))[:frames_needed]

        # If frames fewer than needed, repeat last
        if len(frame_paths) < frames_needed:
            if frame_paths:
                last = frame_paths[-1]
                for k in range(len(frame_paths), frames_needed):
                    dup = tmp / f"dup_{idx}_{k}.png"
                    shutil.copy(last, dup)
                    frame_paths.append(dup)
            else:
                # fallback: create a text image for scene
                txt_img = Image.new("RGB", FRAME_SIZE, (10,10,10))
                draw = ImageDraw.Draw(txt_img)
                draw.text((20, 20), scene, font=default_font, fill=(240,240,240))
                p = tmp / f"scene_{idx}_fallback.png"
                txt_img.save(p)
                frame_paths = [p] * frames_needed

        # Assemble frames to raw scene video
        raw_scene_out = tmp / f"raw_scene_{idx+1}.mp4"
        frames_to_video(frame_paths, str(raw_scene_out), fps=fps, audio_path=None)
        raw_scene_paths.append(raw_scene_out)

    # Concatenate raw scenes
    clips_for_concat = [VideoFileClip(str(p)) for p in raw_scene_paths]
    final_raw = concatenate_videoclips(clips_for_concat, method='compose')
    raw_out = tmp / "final_raw.mp4"
    final_raw.write_videofile(str(raw_out), codec='libx264', fps=fps)

    # Attach audio: prefer user_audio_path, then tts_audio_path
    final_audio = None
    if user_audio_path:
        final_audio = user_audio_path
    elif tts_audio_path:
        final_audio = tts_audio_path

    if final_audio:
        # mux audio
        final_with_audio = tmp / "final_raw_audio.mp4"
        cmd = ["ffmpeg", "-y", "-i", str(raw_out), "-i", final_audio, "-c:v", "copy", "-c:a", "aac", str(final_with_audio)]
        subprocess.check_call(cmd)
        final_raw_path = str(final_with_audio)
    else:
        final_raw_path = str(raw_out)

    # Stylize final video frame-by-frame (CPU fallback faster approach: stylize keyframes or downscale)
    stylized_out = tmp / "final_anime.mp4"

    cap = cv2.VideoCapture(final_raw_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(str(stylized_out), fourcc, fps, (w, h))

    # If ONNX available, use session
    for_frame_apply = apply_animegan_onnx_frame if onnx_session else None
    if apply_anime and onnx_session:
        session = onnx_session
    else:
        session = None

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if apply_anime and session is not None:
            try:
                outf = apply_animegan_onnx_frame(frame, session)
            except Exception as e:
                st.warning(f"ONNX transform failed at frame {frame_i}: {e}. Falling back to OpenCV cartoon")
                outf = cartoonize_image_cv(frame)
        elif apply_anime and use_opencv_fallback:
            outf = cartoonize_image_cv(frame)
        else:
            outf = frame
        out_vid.write(outf)
        frame_i += 1

    cap.release()
    out_vid.release()

    return str(raw_out), str(stylized_out)


# ---------- Streamlit UI: Inputs + Run ----------

st.header("Input")
script_input = st.text_area("Paste your scene-by-scene script here (use 'Scene X:' headings).", height=300)

st.markdown("*Audio:* Upload an audio file to include in the final video, or check TTS in settings.")
user_audio = st.file_uploader("Upload audio (optional)", type=["mp3","wav","m4a"])

col1, col2 = st.columns([1,1])
with col1:
    out_filename = st.text_input("Output filename", value="horror_anime_final.mp4")
with col2:
    allow_preview = st.checkbox("Show preview in app after generation", value=True)

if st.button("Start Generate Video"):
    if not script_input.strip():
        st.error("Please paste your script first.")
        st.stop()

    workdir = tempfile.mkdtemp(prefix="txt2vid_")
    st.info(f"Working in {workdir} — generation may take several minutes (or longer if using local diffusers)")

    # Save uploaded audio if present
    user_audio_path = None
    if user_audio is not None:
        tmp_audio = Path(workdir) / "user_audio" + Path(user_audio.name).suffix
        with open(tmp_audio, "wb") as f:
            f.write(user_audio.read())
        user_audio_path = str(tmp_audio)

    # run pipeline
    try:
        raw_path, anime_path = process_script_to_video(script_input,
                                                        work_dir=workdir,
                                                        per_scene_seconds=scene_seconds,
                                                        fps=fps,
                                                        local_sd=use_local_diffusers,
                                                        external_provider_name=external_provider,
                                                        provider_token=(replicate_token or stability_key),
                                                        apply_anime=apply_animegan,
                                                        onnx_model_path=(onnx_path or None),
                                                        tts_audio_path=None,
                                                        user_audio_path=user_audio_path)

        st.success("Generation complete!")
        # Move final anime file to downloads
        out_final = Path.cwd() / out_filename
        shutil.copy(anime_path, out_final)

        if allow_preview:
            st.video(str(anime_path))

        with open(str(out_final), "rb") as f:
            st.download_button("⬇️ Download final MP4", f, file_name=out_filename)

    except NotImplementedError as e:
        st.error(f"Feature not implemented: {e}")
    except Exception as e:
        st.exception(e)