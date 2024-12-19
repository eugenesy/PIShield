import streamlit as st
from pathlib import Path
from datetime import datetime
# from client import send_image


import requests
from face_obfuscation import process_image
from overlay_privacy import process_with_overlay

SERVER_URL = "http://team3.pailab.csie.ntu.edu.tw:5000/upload"

def main():
    st.title("Streamlit Camera Example")

    # Use the camera input widget to capture an image
    image = st.camera_input("Take a picture")

    if image:
        start_time = datetime.now()
        st.subheader("Captured image")
        st.image(image)

        # Save the image
        save_path = Path("captured_image.jpg")  # Define the file name and path
        with open(save_path, "wb") as f:
            f.write(image.getbuffer())  # Save the BytesIO object to file

        # Show the saved file path
        anonymized_image_dict = process_image(save_path)
        if not anonymized_image_dict["success"]:
            return
        anonymized_image_path = anonymized_image_dict["output_path"]


        st.subheader("Anonymized image")
        end_time = datetime.now()
        st.write(f"Elapsed: {end_time - start_time}")
        start_time = datetime.now()
        st.image(anonymized_image_path)
        # return anonymized_image_path # TODO: delete

        with open(anonymized_image_path, 'rb') as img_file:

            files = {'image': img_file}
            response = requests.post(SERVER_URL, files=files)

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                extension = 'jpg' if 'jpeg' in content_type else 'png' if 'png' in content_type else 'bin'

                output_file = f'processed_image.{extension}'
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print("Processed image received and saved as 'processed_image.jpg'")

                processed_image = output_file
            else:
                print(f"Failed to process image: {response.status_code} - {response.text}")
                return

        st.subheader("Detection mask")
        end_time = datetime.now()
        st.write(f"Elapsed: {end_time - start_time}")
        start_time = datetime.now()
        st.image(processed_image)

        anon_detected = process_with_overlay(anonymized_image_path, processed_image)
        if not anon_detected["success"]:
            print(anon_detected)
            return
        anon_detected_image = anon_detected["output_path"]
        st.subheader("Anonymized image with detection mask")
        end_time = datetime.now()
        st.write(f"Elapsed: {end_time - start_time}")
        start_time = datetime.now()
        st.image(anon_detected_image)

        unanon_detected = process_with_overlay(save_path, processed_image)
        if not unanon_detected["success"]:
            return
        unanon_detected_image = unanon_detected["output_path"]
        st.subheader("Original image with detection mask")
        end_time = datetime.now()
        st.write(f"Elapsed: {end_time - start_time}")
        start_time = datetime.now()
        st.image(unanon_detected_image)

if __name__ == '__main__':
    main()
