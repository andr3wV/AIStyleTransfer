# Style Transfer for Intro to AI
This repo holds the entire version control for my style transfer project for my Intro to AI class. It contains a Flask application for performing image style transfer. It uses a web interface to upload a style image and a content image, and applies the style transfer to the content image.

The program will run 5000 iterations and output the resultant ater every 600th execution. A file called "content.jpg" and "style.jpg" must be in the root directory respresenting the main image and the style to apply to it.
Every 600 executions, the program will output the stylized image. 

At its core, the application employs a neural network to merge the artistic style of one image with the content of another, creating a fusion of the two. Utilizing the VGG19 model, a pre-trained neural network, the application extracts and analyzes the key features of both style and content images. This include texture, color patterns, and brushstroke styles in the case of the style image, and the primary shapes, objects, and spatial distribution in the content image.

The style transfer process involves calculating and matching the Gram matrices of the style image to the content image, ensuring that the style patterns are effectively mirrored while maintaining the integrity of the content structure.

## Installation

To get started, clone the repository and install the required packages:

```bash
git clone [repository-url]
cd [repository-directory]
pip install -r requirements.txt
```

# Usage

To run the application:

1. Start the Flask server:
   \```bash
   python app.py
   \```
2. Open your web browser and navigate to `http://localhost:5000`.
3. Upload a style image and a content image.
4. The application will process the images and display the result.

## Files and Directories

- `app.py`: The main Flask application script.
- `web.py`: Contains the logic for style transfer using PyTorch.
- `templates/index.html`: The HTML template for the web interface.
- `requirements.txt`: Lists all the Python dependencies.

## Credits

- jQuery for frontend interactions.
- Flask for the backend framework.
- PyTorch and VGG for the machine learning model.
