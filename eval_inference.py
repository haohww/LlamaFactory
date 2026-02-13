"""
Full evaluation: run inference on the eval split using a checkpoint,
compare predictions vs ground truth, and save structured results to JSON.

Usage:
  python eval_inference.py [--checkpoint CHECKPOINT_DIR] [--output OUTPUT_JSON]
"""
import argparse
import json
import sys
import time
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ─── Defaults ────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT = "/home/haohw/LlamaFactory/saves/qwen3-vl-8b/full/sft-failurev2/checkpoint-2600"
DATASET_PATH = "/home/haohw/LlamaFactory/data/failurev2/datasetv2/train_all.json"
MEDIA_DIR = "/home/haohw/LlamaFactory/data/failurev2/datasetv2"
DEFAULT_OUTPUT = "/home/haohw/LlamaFactory/eval_results.json"
MAX_NEW_TOKENS = 256
VIDEO_MAX_PIXELS = 57600
VIDEO_MIN_PIXELS = 784
VIDEO_FPS = 2.0
VAL_SIZE = 0.05
SEED = 42
# ─────────────────────────────────────────────────────────────────


def get_eval_split(dataset_path, val_size, seed):
    """Reproduce the exact eval split that LlamaFactory uses."""
    with open(dataset_path) as f:
        data = json.load(f)
    ds = Dataset.from_list(data)
    split = ds.train_test_split(test_size=val_size, seed=seed)
    eval_data = [split["test"][i] for i in range(len(split["test"]))]
    return eval_data


def run_inference(model, processor, sample, media_dir):
    """Run inference on a single sample and return the prediction."""
    user_content = sample["messages"][0]["content"]
    video_path = f"{media_dir}/{sample['videos'][0]}"
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
    return prediction


def classify_result(text):
    """Extract SUCCESS/FAIL from model output."""
    if "SUCCESS" in text:
        return "SUCCESS"
    elif "FAIL" in text:
        return "FAIL"
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description="Run eval inference on the eval split")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to checkpoint directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save results JSON")
    args = parser.parse_args()

    # 1) Load eval split
    print(f"Loading eval split from: {DATASET_PATH}")
    print(f"  val_size={VAL_SIZE}, seed={SEED}")
    eval_data = get_eval_split(DATASET_PATH, VAL_SIZE, SEED)
    print(f"  Eval samples: {len(eval_data)}")
    sys.stdout.flush()

    # 2) Load base model
    print(f"\nLoading base model: {BASE_MODEL}")
    sys.stdout.flush()
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3) Overlay checkpoint weights
    print(f"Loading checkpoint: {args.checkpoint}")
    sys.stdout.flush()
    ckpt_state = load_file(f"{args.checkpoint}/model.safetensors")
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    print(f"  Loaded {len(ckpt_state)} tensors (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    print("Model ready.\n")
    sys.stdout.flush()

    # 4) Run inference on all eval samples
    results = []
    counters = {"total": 0, "correct": 0, "success_correct": 0, "success_total": 0,
                "fail_correct": 0, "fail_total": 0}
    start_time = time.time()

    for idx, sample in enumerate(eval_data):
        user_content = sample["messages"][0]["content"]
        ground_truth = sample["messages"][1]["content"]
        video_path = sample["videos"][0]

        try:
            prediction = run_inference(model, processor, sample, MEDIA_DIR)
        except Exception as e:
            print(f"  [{idx+1}/{len(eval_data)}] ERROR: {e}", flush=True)
            prediction = f"ERROR: {e}"

        gt_class = classify_result(ground_truth)
        pred_class = classify_result(prediction)
        match = gt_class == pred_class

        # Store result
        results.append({
            "index": idx,
            "video": video_path,
            "message": user_content,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "gt_class": gt_class,
            "pred_class": pred_class,
            "match": match,
        })

        # Update counters
        counters["total"] += 1
        if match:
            counters["correct"] += 1
        if gt_class == "SUCCESS":
            counters["success_total"] += 1
            if match:
                counters["success_correct"] += 1
        elif gt_class == "FAIL":
            counters["fail_total"] += 1
            if match:
                counters["fail_correct"] += 1

        # Print progress every 10 samples
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta = (len(eval_data) - idx - 1) / rate if rate > 0 else 0
            acc = counters["correct"] / counters["total"] * 100
            print(f"  [{idx+1}/{len(eval_data)}] acc={acc:.1f}% | "
                  f"{match and 'MATCH' or 'MISMATCH'} GT={gt_class} Pred={pred_class} | "
                  f"ETA={eta/60:.1f}min", flush=True)

    elapsed = time.time() - start_time

    # 5) Compute summary
    summary = {
        "checkpoint": args.checkpoint,
        "eval_samples": counters["total"],
        "overall_accuracy": counters["correct"] / counters["total"] if counters["total"] > 0 else 0,
        "overall_correct": counters["correct"],
        "success_accuracy": counters["success_correct"] / counters["success_total"] if counters["success_total"] > 0 else 0,
        "success_correct": counters["success_correct"],
        "success_total": counters["success_total"],
        "fail_accuracy": counters["fail_correct"] / counters["fail_total"] if counters["fail_total"] > 0 else 0,
        "fail_correct": counters["fail_correct"],
        "fail_total": counters["fail_total"],
        "elapsed_seconds": round(elapsed, 1),
    }

    # 6) Save results
    output = {"summary": summary, "results": results}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 7) Print summary
    print(f"\n{'='*80}")
    print(f"EVAL COMPLETE — {args.checkpoint.split('/')[-1]}")
    print(f"{'='*80}")
    print(f"  Samples:          {summary['eval_samples']}")
    print(f"  Overall accuracy: {summary['overall_correct']}/{summary['eval_samples']} ({summary['overall_accuracy']*100:.1f}%)")
    print(f"  SUCCESS accuracy: {summary['success_correct']}/{summary['success_total']} ({summary['success_accuracy']*100:.1f}%)")
    print(f"  FAIL accuracy:    {summary['fail_correct']}/{summary['fail_total']} ({summary['fail_accuracy']*100:.1f}%)")
    print(f"  Time:             {elapsed/60:.1f} min ({elapsed/len(eval_data):.2f}s per sample)")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
