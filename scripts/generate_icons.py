import os
import struct

os.makedirs('src-tauri/icons', exist_ok=True)

# Minimal 1x1 or 32x32 valid ICO file generator
def create_minimal_ico(filepath):
    # ICO Header: Reserved (2 bytes), Type (2 bytes: 1=ICO), Count (2 bytes: 1)
    header = struct.pack('<HHH', 0, 1, 1)
    
    # 32x32 32-bit BMP image data inside ICO
    width = 32
    height = 32
    bpp = 32
    image_size = 40 + (width * height * 4) + (width * height // 8) # BITMAPINFOHEADER + RGBA + AND mask
    
    # Directory entry (16 bytes)
    directory = struct.pack('<BBBBHHII', width, height, 0, 0, 1, bpp, image_size, 6 + 16)
    
    # BITMAPINFOHEADER (40 bytes)
    bmp_header = struct.pack('<IIIHHIIIIII', 40, width, height * 2, 1, bpp, 0, image_size, 0, 0, 0, 0)
    
    # RGBA Pixel data (32x32 pixels, Cyan glow color 0, 243, 255, 255 -> BGRA format)
    pixel_data = bytearray()
    for _ in range(width * height):
        pixel_data.extend([255, 243, 0, 255]) # B, G, R, A
        
    # AND mask (1 bit per pixel, 0 = opaque)
    and_mask = bytearray(width * height // 8)
    
    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(directory)
        f.write(bmp_header)
        f.write(pixel_data)
        f.write(and_mask)

create_minimal_ico('src-tauri/icons/icon.ico')

# Also write dummy text or dummy png for remaining icon files
for icon_name in ['32x32.png', '128x128.png', '128x128@2x.png', 'icon.icns']:
    path = os.path.join('src-tauri/icons', icon_name)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc` \x05\x00\x00\x04\x00\x01\x04\x02\x0b\xe8\x00\x00\x00\x00IEND\xaeB`\x82')

print("Icons generated successfully!")
