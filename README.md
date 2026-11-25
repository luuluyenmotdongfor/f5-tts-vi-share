[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luuluyenmotdongfor/f5-tts-vi-share/blob/main/colab_tts_vi.ipynb)

# F5-TTS Tiếng Việt — Bộ Giọng Đọc Truyền Cảm (Chạy Colab & Offline)

Kho lưu trữ chia sẻ trọn bộ cấu hình giọng đọc tiếng Việt dùng **F5-TTS** fine-tuned (`hynt/F5-TTS-Vietnamese-ViVoice`), kèm giọng mẫu chuẩn `voice_vi_mau.wav` đã cân chỉnh khớp lời thoại mẫu để sinh giọng đọc tự nhiên, liền mạch.

---

## Cách 1: Chạy trực tiếp trên Google Colab (Khuyên dùng — Không cần máy mạnh)

> **Lưu ý quan trọng trước khi chạy:**
> Trên thanh menu Google Colab, vào **`Runtime`** -> **`Change runtime type`** -> mục **Hardware accelerator** chọn **`T4 GPU`** -> nhấn **Save**. (Nếu dùng CPU sẽ rất chậm).

1. Bấm vào nút **Open in Colab** ở đầu trang.
2. **Cell 1:** Bấm nút Play để Colab tự động tải repo `luuluyenmotdongfor/f5-tts-vi-share`, cài thư viện và nạp sẵn model tiếng Việt lên GPU (khoảng 1–2 phút ở lần đầu).
3. **Cell 2 (Giao diện test trực tiếp):** Bấm nút Play để mở khung nhập kịch bản ngay bên dưới. Bạn có thể dán thoải mái nhiều câu (ngăn cách bằng dấu `|` hoặc xuống dòng), kéo thanh chỉnh tốc độ đọc rồi bấm nút **"🎙️ Tạo giọng đọc & Nghe thử"**.
   - Ngay khi tạo xong, trình phát audio sẽ hiện ra để bạn **nghe thử trực tiếp trên Colab** kèm nút **"⬇️ Tải audio về máy"** (bấm phát tải ngay `.wav` hoặc `.zip` nếu nhiều đoạn).
4. **Cell 3 & 4 (Tuỳ chọn dự phòng):** Dành cho trường hợp muốn chạy nhanh dạng Form `#@param` hoặc tải lại toàn bộ các file đã tạo.

---

## Cách 2: Cài đặt và chạy Offline trên máy cá nhân (Có card đồ họa NVIDIA GPU)

### Bước 1: Tải mã nguồn về máy
```bash
git clone https://github.com/luuluyenmotdongfor/f5-tts-vi-share.git
cd f5-tts-vi-share
```

### Bước 2: Tạo môi trường ảo và cài thư viện
*Yêu cầu: Python 3.10 hoặc 3.11 và Driver NVIDIA đã cài sẵn.*

Trên **Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài PyTorch hỗ trợ GPU NVIDIA (CUDA 12.4) trước
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Cài các thư viện còn lại
pip install -r requirements.txt
```

Trên **Linux / macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio
pip install -r requirements.txt
```

### Bước 3: Chạy sinh giọng đọc
- **Cách nhanh nhất:** Mở file `gen_voice_vi.py`, sửa chữ trong biến `TEXT`, lưu lại rồi chạy:
  ```bash
  python gen_voice_vi.py
  ```
- **Hoặc dùng file text ngoài:** Tạo file `input.txt` đặt cùng thư mục với `gen_voice_vi.py` (mỗi câu cách nhau bằng dấu `|` hoặc xuống dòng) rồi chạy `python gen_voice_vi.py`.
- Kết quả sẽ xuất ra file **`output.wav`** ngay trong thư mục. (Lần chạy đầu tiên máy sẽ tự tải checkpoint `model_last.pt` từ Hugging Face và lưu cache cho các lần sau).

---

## Lưu ý quan trọng để giọng đọc hay và không bị ngọng

Mô hình `hynt/F5-TTS-Vietnamese-ViVoice` đọc chuẩn xác nhất khi văn bản đầu vào tuân thủ 3 quy tắc sau:

1. **Viết toàn bộ con số, ngày tháng, năm thành chữ tiếng Việt:**
   - Không viết: `Năm 1099, có 3.000 kỵ binh lúc 7h30.`
   - Nên viết: `Năm một nghìn không trăm chín mươi chín, có ba nghìn kỵ binh lúc bảy giờ ba mươi.`
2. **Phiên âm mềm tên riêng hoặc từ tiếng Anh sang âm tiếng Việt:**
   - Không viết: `El Cid cưỡi Babieca ở thành Valencia.`
   - Nên viết: `En Xít cưỡi Ba-bi-ê-ca ở thành Va-len-xi-a.`
   - Một số ví dụ quen thuộc: `Video` -> `vi-đê-ô`, `Marketing` -> `ma-két-tinh`, `London` -> `Luân Đôn`.
3. **Ngắt nghỉ câu rõ ràng bằng dấu câu:**
   - Sử dụng dấu phẩy `,` để nghỉ ngắn và dấu chấm `.` / hỏi `?` / than `!` cuối câu.
   - Script đã tự động chuẩn hóa Unicode NFC và chuyển chữ về chữ thường trước khi đưa vào model để đạt độ ổn định cao nhất.
