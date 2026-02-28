import requests

# Your Flask server URL
url = "http://127.0.0.1:5000/predict"

# Replace this with the actual name of an image file on your computer
image_path = "test_image.jpg"

try:
    with open(image_path, "rb") as img_file:
        # We send the image in a dictionary under the key "image" to match your Flask backend
        files = {"image": img_file}
        print(f"Sending {image_path} to FaceShield API...")
        
        response = requests.post(url, files=files)
        
    print("Status Code:", response.status_code)
    print("Result:", response.json())
    
except FileNotFoundError:
    print(f"Error: Could not find '{image_path}'. Make sure the image is in the same folder as this script.")
except Exception as e:
    print("Connection Error:", str(e))