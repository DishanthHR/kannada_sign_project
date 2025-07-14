import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Define the font path
font_path = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\scripts\NotoSansKannada-VariableFont_wdth,wght.ttf"

# Test text
test_text = "ಕನ್ನಡ ಸೈನ್ ಲ್ಯಾಂಗ್ವೇಜ್ ರೆಕಗ್ನಿಷನ್"

def test_kannada_font():
    """Test if the Kannada font renders correctly"""
    try:
        # Create a blank image
        img = np.zeros((300, 800, 3), dtype=np.uint8)
        
        # Load the Kannada font
        font = ImageFont.truetype(font_path, 30)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Draw text
        draw.text((20, 50), test_text, font=font, fill=(255, 255, 255))
        draw.text((20, 100), "ಪ್ರಸ್ತುತ ಪದ: ನಾನು", font=font, fill=(0, 255, 0))
        draw.text((20, 150), "ವಾಕ್ಯ: ನಾನು ಇಲ್ಲಿ ಇದೀಯ", font=font, fill=(255, 0, 0))
        
        # Convert back to OpenCV format
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Display the image
        cv2.imshow('Kannada Font Test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("Kannada font test completed successfully!")
        
    except Exception as e:
        print(f"Error testing Kannada font: {str(e)}")

if __name__ == "__main__":
    test_kannada_font()
