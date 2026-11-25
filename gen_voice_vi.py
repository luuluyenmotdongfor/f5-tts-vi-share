# -*- coding: utf-8 -*-
"""
gen_voice_vi.py — Bộ tạo giọng đọc tiếng Việt bằng F5-TTS (bản rút gọn dễ dùng).

Cách dùng nhanh:
  1. Sửa trực tiếp câu muốn đọc trong biến TEXT bên dưới, HOẶC tạo file input.txt đặt cùng thư mục.
  2. Chạy lệnh: python gen_voice_vi.py
  3. Nhận file kết quả: output.wav
"""
import argparse
import os
import sys
import unicodedata
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from f5_tts.api import F5TTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# CẤU HÌNH NHANH (Sửa nội dung cần đọc ở đây nếu không dùng input.txt)
# Mẹo: Viết số thành chữ tiếng Việt và phiên âm mềm từ tiếng Anh để đọc mượt nhất.
# ==============================================================================
TEXT = """
Xin chào, đây là giọng đọc trí tuệ nhân tạo tiếng Việt chạy bằng mô hình F5 TTS. |
Mùa hè năm một nghìn không trăm chín mươi chín, trước bức tường thành Va-len-xi-a, kỵ sĩ En Xít bất ngờ lao ra giữa ánh bình minh.
"""

# Cấu hình giọng mẫu & model tiếng Việt (đã tinh chỉnh khớp nhau 100%)
REF_WAV = os.path.join(BASE_DIR, "voice_vi_mau.wav")
REF_TEXT = "lấy ngẫu nhiên từng quả ra cho đến khi hết, tính xác suất để lấy được tất cả 4 bóng đỏ, trước khi lấy được 3 bóng trắng. học sinh bình thường nhìn thấy bài này sẽ đâm đầu vào vẽ sơ đồ cây."
CKPT_REPO = "hynt/F5-TTS-Vietnamese-ViVoice"

SPEED = 1.0
SEED = 42
PAUSE_SEC = 0.25


def normalize_vi(text: str) -> str:
    """Chuẩn hóa Unicode NFC, chuyển về chữ thường và dọn khoảng trắng cho F5-TTS."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower().strip()
    text = " ".join(text.split())
    if text and text[-1] not in ".!?":
        text += "."
    return text


def split_segments(raw_text: str):
    """Tách đoạn văn theo dấu | (nếu có) hoặc theo từng dòng."""
    raw_text = raw_text.strip()
    if "|" in raw_text:
        parts = [" ".join(p.split()) for p in raw_text.split("|")]
    else:
        parts = [" ".join(p.split()) for p in raw_text.splitlines()]
    return [p for p in parts if p]


def main():
    ap = argparse.ArgumentParser(description="Tạo giọng đọc tiếng Việt bằng F5-TTS")
    ap.add_argument("--input", default=None, help="Đường dẫn file .txt đầu vào (mặc định tự tìm input.txt)")
    ap.add_argument("--text", default=None, help="Chuỗi văn bản cần đọc trực tiếp")
    ap.add_argument("--output", default=os.path.join(BASE_DIR, "output.wav"), help="File .wav đầu ra")
    ap.add_argument("--ref", default=REF_WAV, help="File giọng mẫu .wav")
    ap.add_argument("--speed", type=float, default=SPEED, help="Tốc độ đọc (mặc định 1.0)")
    ap.add_argument("--seed", type=int, default=SEED, help="Seed cố định chất giọng")
    args = ap.parse_args()

    if not os.path.exists(args.ref):
        sys.exit(f"Không tìm thấy file giọng mẫu: {args.ref}")

    default_input = os.path.join(BASE_DIR, "input.txt")
    if args.text:
        raw_text = args.text
        source_name = "--text"
    elif args.input:
        with open(args.input, encoding="utf-8-sig") as f:
            raw_text = f.read()
        source_name = args.input
    elif os.path.exists(default_input):
        with open(default_input, encoding="utf-8-sig") as f:
            raw_text = f.read()
        source_name = "input.txt"
    else:
        raw_text = TEXT
        source_name = "biến TEXT"

    segments = split_segments(raw_text)
    if not segments:
        sys.exit("Không có nội dung văn bản nào để đọc.")

    print("=" * 60)
    print("F5-TTS TIẾNG VIỆT")
    print(f"Nguồn chữ : {source_name} ({len(segments)} đoạn)")
    print(f"Giọng mẫu : {os.path.basename(args.ref)}")
    print(f"Đầu ra    : {args.output}")
    print("=" * 60)

    print("\n[1/2] Đang kiểm tra / tải checkpoint hynt/F5-TTS-Vietnamese-ViVoice...")
    ckpt_file = hf_hub_download(repo_id=CKPT_REPO, filename="model_last.pt")
    vocab_file = hf_hub_download(repo_id=CKPT_REPO, filename="config.json")

    print("[2/2] Đang nạp model F5TTS_Base lên thiết bị...")
    tts = F5TTS(
        model="F5TTS_Base",
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
    )

    ref_text_norm = normalize_vi(REF_TEXT)
    tmp_wav = os.path.join(BASE_DIR, "_tmp_part.wav")
    chunks = []
    out_sr = 24000

    try:
        for i, seg in enumerate(segments, 1):
            seg_norm = normalize_vi(seg)
            print(f"[{i}/{len(segments)}] Đang đọc: {seg_norm}")
            tts.infer(
                ref_file=args.ref,
                ref_text=ref_text_norm,
                gen_text=seg_norm,
                file_wave=tmp_wav,
                speed=args.speed,
                seed=args.seed,
                remove_silence=False,
            )
            data, sr = sf.read(tmp_wav)
            if data.ndim > 1:
                data = data.mean(axis=1)
            out_sr = sr
            chunks.append(data.astype(np.float32))
            if i < len(segments) and PAUSE_SEC > 0:
                chunks.append(np.zeros(int(PAUSE_SEC * sr), dtype=np.float32))
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

    merged = np.concatenate(chunks)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(args.output, merged, out_sr)
    dur = len(merged) / float(out_sr)
    print(f"\n>>> HOÀN TẤT! File âm thanh ({dur:.2f} giây): {args.output}")


if __name__ == "__main__":
    main()
