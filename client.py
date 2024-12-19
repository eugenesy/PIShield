import requests
from face_obfuscation import process_image

SERVER_URL = "http://team3.pailab.csie.ntu.edu.tw:5000/upload"

def send_image(image_path):
    anonymized_image_dict = process_image(image_path)
    if not anonymized_image_dict["success"]:
        return None # TODO
    anonymized_image_path = anonymized_image_dict["output_path"]
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
            return output_file
        else:
            print(f"Failed to process image: {response.status_code} - {response.text}")
            return None

if __name__ == '__main__':
    send_image('image_to_send.jpg')


