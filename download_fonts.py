#!/usr/bin/env python3
"""
download_fonts.py: Tải bộ sưu tập font đa dạng từ Google Fonts và quét system fonts.
Hỗ trợ cả Sans-serif, Serif, Monospace, Display (nghệ thuật) và Handwriting (chữ viết tay).
Tự động lọc bỏ triệt để các font rác hoặc font cổ ngữ vẽ ô vuông tofu ▯.
"""

import os
import glob
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageFont, Image, ImageDraw
import numpy as np
from tqdm import tqdm

GOOGLE_FONTS = [
    # --- Sans-serif (Hiện đại, rõ ràng, tech) ---
    ("Roboto", "ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf"),
    ("OpenSans", "ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf"),
    ("Montserrat", "ofl/montserrat/Montserrat%5Bwght%5D.ttf"),
    ("Lato", "ofl/lato/Lato-Regular.ttf"),
    ("Poppins", "ofl/poppins/Poppins-Regular.ttf"),
    ("Oswald", "ofl/oswald/Oswald%5Bwght%5D.ttf"),
    ("Raleway", "ofl/raleway/Raleway%5Bwght%5D.ttf"),
    ("Nunito", "ofl/nunito/Nunito%5Bwght%5D.ttf"),
    ("Anton", "ofl/anton/Anton-Regular.ttf"),
    ("Barlow", "ofl/barlow/Barlow-Regular.ttf"),
    ("BebasNeue", "ofl/bebasneue/BebasNeue-Regular.ttf"),
    ("Teko", "ofl/teko/Teko%5Bwght%5D.ttf"),
    ("Rubik", "ofl/rubik/Rubik%5Bwght%5D.ttf"),
    ("Kanit", "ofl/kanit/Kanit-Regular.ttf"),
    ("Comfortaa", "ofl/comfortaa/Comfortaa%5Bwght%5D.ttf"),

    # --- Serif (Cổ điển, sách báo, trang trọng) ---
    ("PlayfairDisplay", "ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf"),
    ("Merriweather", "ofl/merriweather/Merriweather%5Bopsz%2Cwdth%2Cwght%5D.ttf"),
    ("Lora", "ofl/lora/Lora%5Bwght%5D.ttf"),
    ("PTSerif", "ofl/ptserif/PT_Serif-Web-Regular.ttf"),
    ("Cinzel", "ofl/cinzel/Cinzel%5Bwght%5D.ttf"),
    ("CinzelDecorative", "ofl/cinzeldecorative/CinzelDecorative-Regular.ttf"),
    ("CormorantGaramond", "ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf"),
    ("Arvo", "ofl/arvo/Arvo-Regular.ttf"),
    ("Bitter", "ofl/bitter/Bitter%5Bwght%5D.ttf"),
    ("EBGaramond", "ofl/ebgaramond/EBGaramond%5Bwght%5D.ttf"),

    # --- Handwriting / Cursive (Chữ viết tay, thư pháp tự nhiên) ---
    ("Caveat", "ofl/caveat/Caveat%5Bwght%5D.ttf"),
    ("Pacifico", "ofl/pacifico/Pacifico-Regular.ttf"),
    ("DancingScript", "ofl/dancingscript/DancingScript%5Bwght%5D.ttf"),
    ("IndieFlower", "ofl/indieflower/IndieFlower-Regular.ttf"),
    ("ShadowsIntoLight", "ofl/shadowsintolight/ShadowsIntoLight.ttf"),
    ("Kalam", "ofl/kalam/Kalam-Regular.ttf"),
    ("PatrickHand", "ofl/patrickhand/PatrickHand-Regular.ttf"),
    ("GloriaHallelujah", "ofl/gloriahallelujah/GloriaHallelujah.ttf"),
    ("Yellowtail", "apache/yellowtail/Yellowtail-Regular.ttf"),
    ("Satisfy", "apache/satisfy/Satisfy-Regular.ttf"),
    ("Sacramento", "ofl/sacramento/Sacramento-Regular.ttf"),
    ("Courgette", "ofl/courgette/Courgette-Regular.ttf"),
    ("MarckScript", "ofl/marckscript/MarckScript-Regular.ttf"),
    ("GreatVibes", "ofl/greatvibes/GreatVibes-Regular.ttf"),
    ("AlexBrush", "ofl/alexbrush/AlexBrush-Regular.ttf"),
    ("Allura", "ofl/allura/Allura-Regular.ttf"),
    ("BadScript", "ofl/badscript/BadScript-Regular.ttf"),
    ("KaushanScript", "ofl/kaushanscript/KaushanScript-Regular.ttf"),
    ("Lobster", "ofl/lobster/Lobster-Regular.ttf"),

    # --- Display / Artistic / Comic / Retro (Biển báo, hoạt hình, game, nghệ thuật) ---
    ("Bangers", "ofl/bangers/Bangers-Regular.ttf"),
    ("Bungee", "ofl/bungee/Bungee-Regular.ttf"),
    ("PressStart2P", "ofl/pressstart2p/PressStart2P-Regular.ttf"),
    ("SpecialElite", "apache/specialelite/SpecialElite-Regular.ttf"),
    ("Creepster", "ofl/creepster/Creepster-Regular.ttf"),
    ("Monoton", "ofl/monoton/Monoton-Regular.ttf"),
    ("FasterOne", "ofl/fasterone/FasterOne-Regular.ttf"),
    ("Fredoka", "ofl/fredoka/Fredoka%5Bwdth%2Cwght%5D.ttf"),
    ("Audiowide", "ofl/audiowide/Audiowide-Regular.ttf"),
    ("Orbitron", "ofl/orbitron/Orbitron%5Bwght%5D.ttf"),
    ("BlackOpsOne", "ofl/blackopsone/BlackOpsOne-Regular.ttf"),
    # --- Monospace (Code, máy đánh chữ) ---
    ("FiraCode", "ofl/firacode/FiraCode%5Bwght%5D.ttf"),
    ("SourceCodePro", "ofl/sourcecodepro/SourceCodePro%5Bwght%5D.ttf"),
    ("SpaceMono", "ofl/spacemono/SpaceMono-Regular.ttf"),
    ("Inconsolata", "ofl/inconsolata/Inconsolata%5Bwdth%2Cwght%5D.ttf"),
    ("CourierPrime", "ofl/courierprime/CourierPrime-Regular.ttf"),
]

BASE_URL = "https://raw.githubusercontent.com/google/fonts/main/"


def is_valid_font(font_path: str) -> bool:
    """Kiểm tra font có thực sự render dc chữ cái Latin & số kh (loại bỏ triệt để font rác/tofu ▯/ornaments)."""
    fname = os.path.basename(font_path).lower()
    if any(k in fname for k in ["ornament", "symbol", "braille", "math", "dingbat", "emoj"]):
        return False
    try:
        font = ImageFont.truetype(font_path, size=32)
        # Tạo ảnh mẫu của kí tự không tồn tại (\uffff) để lấy hình dáng của ô vuông tofu (.notdef)
        img_missing = Image.new("L", (40, 40), 0)
        draw = ImageDraw.Draw(img_missing)
        draw.text((5, 5), "\uffff", font=font, fill=255)
        missing_arr = np.array(img_missing)
        has_notdef_glyph = np.count_nonzero(missing_arr) > 0

        # Kiểm tra các kí tự đại diện: HOA, thường, số, nét phức tạp
        for c in ["A", "Z", "a", "z", "0", "9", "g", "Q", "W", "1"]:
            img_c = Image.new("L", (40, 40), 0)
            d_c = ImageDraw.Draw(img_c)
            d_c.text((5, 5), c, font=font, fill=255)
            arr_c = np.array(img_c)
            if np.count_nonzero(arr_c) == 0:
                return False
            # Nếu giống hệt ô vuông tofu -> font không hỗ trợ Latin!
            if has_notdef_glyph and np.array_equal(arr_c, missing_arr):
                return False
        return True
    except Exception:
        return False


def download_one_font(item, target_dir):
    name, rel_path = item
    dest_path = os.path.join(target_dir, f"{name}.ttf")
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return dest_path, True

    url = BASE_URL + rel_path
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
            if len(content) > 1000:
                with open(dest_path, "wb") as f:
                    f.write(content)
                if is_valid_font(dest_path):
                    return dest_path, True
                else:
                    os.remove(dest_path)
                    return dest_path, False
    except Exception:
        pass
    return dest_path, False


def scan_system_fonts(target_dir):
    """Quét thêm font có sẵn trong hệ điều hành (macOS & Linux)."""
    search_dirs = [
        "/System/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        "/usr/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
    ]

    copied = 0
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for ext in ("*.ttf", "*.otf"):
            for font_file in glob.glob(os.path.join(sdir, ext)):
                fname = os.path.basename(font_file)
                dest = os.path.join(target_dir, f"sys_{fname}")
                if not os.path.exists(dest) and is_valid_font(font_file):
                    try:
                        shutil.copyfile(font_file, dest)
                        copied += 1
                    except Exception:
                        pass
    return copied


def main():
    fonts_dir = os.path.abspath("./fonts")
    os.makedirs(fonts_dir, exist_ok=True)

    print(f"[*] Đang tải {len(GOOGLE_FONTS)} Google Fonts chất lượng cao...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_one_font, item, fonts_dir) for item in GOOGLE_FONTS]
        success_count = sum(1 for f in tqdm(futures, desc="Downloading Google Fonts") if f.result()[1])

    print(f"[+] Đã tải thành công {success_count}/{len(GOOGLE_FONTS)} Google Fonts vào '{fonts_dir}'.")

    print("[*] Đang quét và bổ sung font có sẵn trong hệ điều hành...")
    sys_count = scan_system_fonts(fonts_dir)
    print(f"[+] Đã bổ sung thêm {sys_count} fonts từ hệ thống.")

    total_valid = [f for f in glob.glob(os.path.join(fonts_dir, "*")) if is_valid_font(f)]
    print(f"[✓] TỔNG CỘNG: Đã sẵn sàng {len(total_valid)} fonts đa dạng để sinh dataset!")


if __name__ == "__main__":
    main()
