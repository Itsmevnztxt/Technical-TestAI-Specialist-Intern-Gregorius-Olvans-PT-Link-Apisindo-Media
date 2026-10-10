import os
import sys
import cv2
import easyocr
import numpy as np

def get_text_color(r, g, b):
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return "black" if luminance > 128 else "white"

def get_dominant_bg_color(img, x, y, w, h):
    """
    Mengambil sampel warna latar belakang yang lebih akurat
    dengan menganalisis margin di sekitar kotak teks (bukan cuma 1 piksel).
    """
    img_h, img_w, _ = img.shape
    y1 = max(0, y - 4)
    y2 = min(img_h, y + h + 4)
    x1 = max(0, x - 4)
    x2 = min(img_w, x + w + 4)
    
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "rgb(255, 255, 255)", "black"

    median_color = np.median(crop.reshape(-1, 3), axis=0)
    b, g, r = int(median_color[0]), int(median_color[1]), int(median_color[2])
    
    bg_color_css = f"rgb({r}, {g}, {b})"
    text_color_css = get_text_color(r, g, b)
    return bg_color_css, text_color_css

def main(image_path):
    if not os.path.exists(image_path):
        print(f"[-] Error: File '{image_path}' tidak ditemukan.")
        return

    print(f"[*] Membaca citra: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("[-] Error: OpenCV gagal membaca gambar.")
        return

    img_height, img_width, _ = img.shape
    
    print("[*] Inisialisasi EasyOCR Pipeline...")
    reader = easyocr.Reader(['id', 'en'], gpu=False)
    results = reader.readtext(image_path)
    
    print(f"[*] Memproses {len(results)} blok teks dan menyusun HTML...")

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>OCR Extraction Result - Interactive Slide</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            background-color: #121212;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0;
            padding: 30px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #fff;
        }}
        .header-info {{
            margin-bottom: 20px;
            text-align: center;
        }}
        .badge {{
            background: #007bff;
            color: #fff;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: bold;
        }}
        .slide-container {{
            position: relative;
            width: {img_width}px;
            height: {img_height}px;
            background-image: url('{image_path}');
            background-size: 100% 100%;
            box-shadow: 0px 15px 35px rgba(0, 0, 0, 0.7);
            border-radius: 8px;
            overflow: hidden;
        }}
        .text-box {{
            position: absolute;
            display: flex;
            align-items: center;
            white-space: nowrap;
            overflow: hidden;
            padding: 2px 4px;
            border-radius: 3px;
            /* Garis bantu halus agar terlihat jelas bahwa ini elemen HTML */
            border: 1px dashed rgba(0, 123, 255, 0.5);
            transition: all 0.2s ease-in-out;
            cursor: text;
        }}
        .text-box:hover {{
            border: 2px solid #00ff00 !important;
            transform: scale(1.02);
            z-index: 100 !important;
            box-shadow: 0 4px 10px rgba(0, 255, 0, 0.4);
        }}
        .text-box:focus {{
            outline: none;
            border: 2px solid #ff9800 !important;
            z-index: 101 !important;
        }}
    </style>
</head>
<body>
    <div class="header-info">
        <h2>Interactive OCR Document Reconstruction</h2>
        <span class="badge">Total Blok Teks: {len(results)} | Contenteditable Enabled</span>
    </div>

    <div class="slide-container">
"""
    
    for idx, (bbox, text, prob) in enumerate(results):
        (tl, tr, br, bl) = bbox
        x = int(min(tl[0], bl[0]))
        y = int(min(tl[1], tr[1]))
        w = int(max(tr[0], br[0]) - x)
        h = int(max(br[1], bl[1]) - y)

        bg_color, text_color = get_dominant_bg_color(img, x, y, w, h)

        font_size = max(12, int(h * 0.72))
        
        html_content += f"""        <span class="text-box" 
              contenteditable="true" 
              title="Confidence: {prob:.2%} (Klik untuk mengedit)"
              style="left: {x}px; top: {y}px; width: {w}px; height: {h}px; font-size: {font_size}px; background-color: {bg_color}; color: {text_color}; z-index: {idx + 1};">
            {text}
        </span>
"""
        
    html_content += """    </div>
</body>
</html>
"""
    
    output_filename = "output.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"[+] Berhasil! File tersimpan di '{output_filename}'. Buka di browser untuk melihat hasilnya.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "slide.jpg"
    main(target)