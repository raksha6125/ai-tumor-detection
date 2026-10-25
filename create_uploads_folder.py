# create_uploads_folder.py
import os

def setup_upload_folder():
    base_path = os.path.join('static', 'uploads')
    os.makedirs(base_path, exist_ok=True)
    print(f"✅ Uploads folder ready at: {base_path}")

if __name__ == "__main__":
    setup_upload_folder()
