"""
Batch smoke test: run inference on training samples across multiple checkpoints.
"""
import json
import random
import sys
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ─── Configuration ───────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CHECKPOINTS = [
    "/home/haohw/LlamaFactory/saves/qwen3-vl-8b/full/sft-failurev1/checkpoint-600",
    "/home/haohw/LlamaFactory/saves/qwen3-vl-8b/full/sft-failurev1/checkpoint-650",
    "/home/haohw/LlamaFactory/saves/qwen3-vl-8b/full/sft-failurev1/checkpoint-700",
]
DATASET_PATH = "/home/haohw/LlamaFactory/dataset/datasetv1/train_all.json"
MEDIA_DIR = "/home/haohw/LlamaFactory/dataset/datasetv1"
NUM_SAMPLES = 10
MAX_NEW_TOKENS = 256
VIDEO_MAX_PIXELS = 57600
VIDEO_MIN_PIXELS = 784
VIDEO_FPS = 2.0
SEED = 42
# ─────────────────────────────────────────────────────────────────

random.seed(SEED)

# Load dataset and select samples (same across all checkpoints for fair comparison)
with open(DATASET_PATH) as f:
    dataset = json.load(f)

success_samples = [s for s in dataset if "SUCCESS" in s["messages"][1]["content"]]
fail_samples = [s for s in dataset if "FAIL" in s["messages"][1]["content"]]
n_each = NUM_SAMPLES // 2
selected = random.sample(success_samples, min(n_each, len(success_samples))) + \
           random.sample(fail_samples, min(n_each, len(fail_samples)))
random.shuffle(selected)

print(f"Selected {len(selected)} samples ({n_each} SUCCESS + {n_each} FAIL)")
print(f"Loading base model: {BASE_MODEL}")
sys.stdout.flush()

# Load base model once
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

all_summaries = []

for ckpt_dir in CHECKPOINTS:
    ckpt_name = ckpt_dir.split("/")[-1]
    print(f"\n{'#'*100}")
    print(f"# TESTING: {ckpt_name}")
    print(f"{'#'*100}")
    sys.stdout.flush()

    # Load checkpoint weights
    ckpt_state = load_file(f"{ckpt_dir}/model.safetensors")
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    print(f"  Loaded {len(ckpt_state)} tensors (missing={len(missing)}, unexpected={len(unexpected)})")
    sys.stdout.flush()
    model.eval()

    results = []
    for idx, sample in enumerate(selected):
        user_content = sample["messages"][0]["content"]
        ground_truth = sample["messages"][1]["content"]
        video_path = f"{MEDIA_DIR}/{sample['videos'][0]}"
        text_part = user_content.replace("<video>", "").strip()

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

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
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

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()

        gt_result = "SUCCESS" if "SUCCESS" in ground_truth else "FAIL"
        pred_result = "SUCCESS" if "SUCCESS" in prediction else ("FAIL" if "FAIL" in prediction else "UNKNOWN")
        match = "MATCH" if gt_result == pred_result else "MISMATCH"
        results.append({"match": gt_result == pred_result, "gt_result": gt_result})

        print(f"\n  [{idx+1}/{len(selected)}] {match} | GT={gt_result} Pred={pred_result}")
        print(f"    GT:   {ground_truth[:200]}")
        print(f"    Pred: {prediction[:200]}")
        sys.stdout.flush()

    total = len(results)
    correct = sum(1 for r in results if r["match"])
    s_correct = sum(1 for r in results if r["match"] and r["gt_result"] == "SUCCESS")
    f_correct = sum(1 for r in results if r["match"] and r["gt_result"] == "FAIL")
    s_total = sum(1 for r in results if r["gt_result"] == "SUCCESS")
    f_total = sum(1 for r in results if r["gt_result"] == "FAIL")

    summary = {
        "checkpoint": ckpt_name,
        "overall": f"{correct}/{total} ({100*correct/total:.0f}%)",
        "success": f"{s_correct}/{s_total}",
        "fail": f"{f_correct}/{f_total}",
    }
    all_summaries.append(summary)

    print(f"\n  --- {ckpt_name} RESULT: Overall={summary['overall']} SUCCESS={summary['success']} FAIL={summary['fail']} ---")
    sys.stdout.flush()

# Final comparison table
print(f"\n\n{'='*100}")
print(f"{'COMPARISON TABLE':^100}")
print(f"{'='*100}")
print(f"{'Checkpoint':<20} {'Overall':<20} {'SUCCESS':<15} {'FAIL':<15}")
print(f"{'-'*70}")
for s in all_summaries:
    print(f"{s['checkpoint']:<20} {s['overall']:<20} {s['success']:<15} {s['fail']:<15}")
print(f"{'='*100}")
