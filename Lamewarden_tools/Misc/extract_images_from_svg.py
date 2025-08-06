from bs4 import BeautifulSoup
import base64
import os
import requests

def extract_images_from_svg(svg_path, output_dir):
    with open(svg_path, 'r') as file:
        soup = BeautifulSoup(file, 'lxml')
    
    images = soup.find_all('image')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, image in enumerate(images):
        href = image.get('xlink:href') or image.get('href')
        if not href:
            continue
        
        if href.startswith('data:image'):
            image_type, image_data = href.split(',')
            image_ext = image_type.split('/')[1].split(';')[0]
            image_data = base64.b64decode(image_data)
            
            image_path = os.path.join(output_dir, f'image_{i}.{image_ext}')
            with open(image_path, 'wb') as img_file:
                img_file.write(image_data)
        else:
            # Handling externally linked images (URLs or file paths)
            try:
                if href.startswith('http'):
                    response = requests.get(href)
                    response.raise_for_status()
                    image_ext = href.split('.')[-1]
                    image_data = response.content
                else:
                    with open(href, 'rb') as img_file:
                        image_data = img_file.read()
                    image_ext = href.split('.')[-1]
                
                image_path = os.path.join(output_dir, f'image_{i}.{image_ext}')
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_data)
            except Exception as e:
                print(f"Could not retrieve image from {href}: {e}")

# Usage

# extract_images_from_svg(r"C:\PSI\Conferences\2024.07.SEB\2024.07.SEB_poster_Kashkan.svg", r"C:\PSI\Conferences\2024.07.SEB\Outputs")