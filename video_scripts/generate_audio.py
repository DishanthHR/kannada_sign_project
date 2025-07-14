import os
import sys
from gtts import gTTS
import pygame

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
audio_dir = os.path.join(base_dir, 'audio')

# Create audio directory if it doesn't exist
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)
    print(f"Created audio directory at {audio_dir}")

# Define the Kannada words
kannada_words = [
    "ಆಡು", "ಇದೀಯ", "ಇಲ್ಲ", "ಇಲ್ಲಿ", "ಇವತ್ತು", "ಎಲ್ಲಿ", "ಏಕೆ", "ಏನು", 
    "ಕುಳಿತುಕೊ", "ಕೇಳು", "ಗಲಾಟೆ", "ಜೊತೆ", "ತಿಂದೆ", "ತೆಗೆದುಕೋ", "ನಾನು", 
    "ನಿಧಾನವಾಗಿ", "ನಿನ್ನ", "ನೀನು", "ಪುಸ್ತಕ", "ಬಂದರು", "ಮಾಡಬೇಡಿ", "ಮಾತು", 
    "ಯಾರು", "ವಾಸವಾಗಿ", "ಸುಮ್ಮನೆ", "ಹೆಚ್ಚು", "ಹೋಗಿದ್ದೆ"
]

def generate_audio_files():
    """Generate audio files for each Kannada word"""
    print("Generating audio files for Kannada words...")
    
    # Initialize pygame for audio playback
    pygame.mixer.init()
    
    for word in kannada_words:
        audio_path = os.path.join(audio_dir, f"{word}.mp3")
        
        # Skip if file already exists
        if os.path.exists(audio_path):
            print(f"Audio file for '{word}' already exists. Skipping.")
            continue
        
        try:
            # Generate TTS audio
            tts = gTTS(text=word, lang='kn', slow=False)
            tts.save(audio_path)
            print(f"Generated audio for '{word}'")
            
            # Play the audio for verification
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            pygame.time.wait(1000)  # Wait for 1 second
            
        except Exception as e:
            print(f"Error generating audio for '{word}': {str(e)}")
    
    print("Audio generation complete!")

if __name__ == "__main__":
    generate_audio_files()
