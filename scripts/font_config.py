import matplotlib.font_manager as fm
import matplotlib as mpl
import os
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def setup_kannada_fonts():
    """Set up Kannada font for matplotlib using your existing font file"""
    # Path to your Noto Sans Kannada font
    font_path = os.path.join(os.path.dirname(__file__), 
                            "NotoSansKannada-VariableFont_wdth,wght.ttf")
    
    # Check if the font file exists
    if not os.path.exists(font_path):
        # Try the full path you provided
        font_path = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\scripts\NotoSansKannada-VariableFont_wdth,wght.ttf"
        
        if not os.path.exists(font_path):
            print("Kannada font file not found. Using fallback method.")
            return None
    
    # Register the font with matplotlib
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.ttflist.append(fm.FontEntry(
        fname=font_path,
        name='Noto Sans Kannada',
        style='normal',
        variant='normal',
        weight='normal',
        stretch='normal',
        size='medium'
    ))
    
    # Set as default font
    mpl.rcParams['font.family'] = 'Noto Sans Kannada'
    print(f"Using Kannada font: {font_path}")
    return font_path  # Return the path instead of True
    
def get_kannada_class_names():
    """Return the list of Kannada words for signs"""
    # Your list of Kannada words
    return ["ಆಡು", "ಇದೀಯ", "ಇಲ್ಲ", "ಇಲ್ಲಿ", "ಇವತ್ತು", "ಎಲ್ಲಿ", "ಎಲ್ಲಿಗೆ", 
            "ಏಕೆ", "ಏನು", "ಕುಳಿತುಕೊ", "ಕೇಳು", "ಗಲಾಟೆ", "ಜೊತೆ", "ತಿಂದೆ", 
            "ತೆಗೆದುಕೋ", "ನಾನು", "ನಿಧವಾಗಿ", "ನಿನ್ನ", "ನೀನು", "ಪುಸ್ತಕ", 
            "ಬಂದರು", "ಮಾಡಬೇಡಿ", "ಮಾತು", "ಯಾರು", "ವಾಸವಾಗಿ", "ಸುಮ್ಮನೆ", 
            "ಹೆಚ್ಚು", "ಹೋಗಿದ್ದೆ"]

def transliterate_kannada_to_english(text):
    """Convert Kannada text to English transliteration"""
    if isinstance(text, list):
        return [transliterate_kannada_to_english(item) for item in text]
    
    try:
        # Try to transliterate from Kannada to Latin
        return transliterate(text, sanscript.KANNADA, sanscript.ITRANS)
    except:
        # If transliteration fails, return the original text
        return text

def get_english_class_names():
    """Get English transliterations of the Kannada class names"""
    kannada_names = get_kannada_class_names()
    return transliterate_kannada_to_english(kannada_names)

# Test the function if this file is run directly
if __name__ == "__main__":
    success = setup_kannada_fonts()
    print(f"Font setup successful: {success}")
    
    kannada_names = get_kannada_class_names()
    english_names = get_english_class_names()
    
    print("Kannada class names:")
    print(kannada_names)
    print("\nEnglish transliterations:")
    print(english_names)
