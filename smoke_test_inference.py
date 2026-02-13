"""
Smoke test: run inference on a few training samples using a checkpoint
and compare predictions vs ground truth.
"""
import json
import random
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ─── Configuration ───────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CHECKPOINT_DIR = "/home/haohw/LlamaFactory/saves/qwen3-vl-8b/full/sft-failurev1/checkpoint-550"
DATASET_PATH = "/home/haohw/LlamaFactory/dataset/datasetv1/train_all.json"
MEDIA_DIR = "/home/haohw/LlamaFactory/dataset/datasetv1"
NUM_SAMPLES = 10
MAX_NEW_TOKENS = 256
VIDEO_MAX_PIXELS = 57600
VIDEO_MIN_PIXELS = 784       # 28*28, minimum for Qwen VL
VIDEO_FPS = 2.0
SEED = 42
# ─────────────────────────────────────────────────────────────────

random.seed(SEED)

# Load dataset
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# Pick a balanced sample: half SUCCESS, half FAIL
success_samples = [s for s in dataset if "SUCCESS" in s["messages"][1]["content"]]
fail_samples = [s for s in dataset if "FAIL" in s["messages"][1]["content"]]

n_each = NUM_SAMPLES // 2
selected = random.sample(success_samples, min(n_each, len(success_samples))) + \
           random.sample(fail_samples, min(n_each, len(fail_samples)))
random.shuffle(selected)

print(f"Selected {len(selected)} samples ({n_each} SUCCESS + {n_each} FAIL)")

# 1) Load base model (has full weights including vision tower)
print(f"Loading base model: {BASE_MODEL}")
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 2) Overlay fine-tuned language model weights from checkpoint
print(f"Loading checkpoint weights from: {CHECKPOINT_DIR}")
ckpt_state = load_file(f"{CHECKPOINT_DIR}/model.safetensors")
missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
print(f"  Loaded {len(ckpt_state)} tensors from checkpoint")
print(f"  Missing (vision/projector, expected): {len(missing)}")
print(f"  Unexpected: {len(unexpected)}")

model.eval()
print("Model ready. Running inference...\n")
print("=" * 100)

results = []
for idx, sample in enumerate(selected):
    user_content = sample["messages"][0]["content"]
    ground_truth = sample["messages"][1]["content"]
    video_path = f"{MEDIA_DIR}/{sample['videos'][0]}"

    # Extract text instruction (after <video> tag)
    text_part = user_content.replace("<video>", "").strip()

    # Build messages for the processor
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{video_path}",
                 "max_pixels": VIDEO_MAX_PIXELS,
                 "min_pixels": VIDEO_MIN_PIXELS,
                 "fps": VIDEO_FPS},
                {"type": "text", "text": text_part},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    # Unwrap fps from list to scalar (process_vision_info returns {'fps': [2.0]})
    fps_val = video_kwargs.get("fps", [VIDEO_FPS])
    if isinstance(fps_val, list):
        fps_val = fps_val[0] if fps_val else VIDEO_FPS
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        fps=fps_val,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    # Decode only the new tokens
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()

    # Compare
    gt_result = "SUCCESS" if "SUCCESS" in ground_truth else "FAIL"
    pred_result = "SUCCESS" if "SUCCESS" in prediction else ("FAIL" if "FAIL" in prediction else "UNKNOWN")
    match = "MATCH" if gt_result == pred_result else "MISMATCH"

    results.append({"match": gt_result == pred_result, "gt_result": gt_result})

    print(f"\n[Sample {idx+1}/{len(selected)}] Video: {sample['videos'][0]}")
    print(f"  GT Result:   {gt_result}")
    print(f"  Pred Result: {pred_result}  [{match}]")
    print(f"  GT Full:     {ground_truth[:300]}")
    print(f"  Pred Full:   {prediction[:300]}")
    print("-" * 100)

# Summary
total = len(results)
correct = sum(1 for r in results if r["match"])
success_correct = sum(1 for r in results if r["match"] and r["gt_result"] == "SUCCESS")
fail_correct = sum(1 for r in results if r["match"] and r["gt_result"] == "FAIL")
success_total = sum(1 for r in results if r["gt_result"] == "SUCCESS")
fail_total = sum(1 for r in results if r["gt_result"] == "FAIL")

print(f"\n{'=' * 100}")
print(f"SMOKE TEST SUMMARY")
print(f"  Overall accuracy:  {correct}/{total} ({100*correct/total:.0f}%)")
print(f"  SUCCESS accuracy:  {success_correct}/{success_total}")
print(f"  FAIL accuracy:     {fail_correct}/{fail_total}")
print(f"{'=' * 100}")
