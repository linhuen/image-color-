from flask import Flask, render_template, request
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_colors(image_path, num_colors=5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(img)

    colors = kmeans.cluster_centers_.astype(int)
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors]

    return hex_colors, colors.tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

            hex_colors, rgb_colors = extract_colors(image_path)

            return render_template('index.html',
                                   image_path=image_path,
                                   hex_colors=hex_colors,
                                   rgb_colors=rgb_colors)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
