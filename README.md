# Clip2ClassDist

Clip2ClassDist is a Python script that analyzes image classes by leveraging OpenAI's CLIP model to generate image embeddings. It computes the center and radius of clusters formed by embeddings, the distribution of dot products within each class, and the average, minimum, and maximum dot products between different classes.

The script takes a list of image folders as input, with each folder representing a distinct image class. It uses the specified CLIP model to create normalized embeddings for each image in the folders. The script then analyzes the embeddings using the Faiss library for efficient similarity search and clustering.

## Installation

Clone the repository:

```bash
git clone https://github.com/ternaus/clip2classdist.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The main script compute.py can be executed with the following command:

```bash
Copy code
python clip2class_dist/compute.py --clip_model MODEL_NAME --image_folders CLASS_NAME_1 FOLDER_PATH_1 CLASS_NAME_2 FOLDER_PATH_2 [--device DEVICE]
```

Arguments:

* --clip_model: Name of the CLIP model to use (e.g., `ViT-B/32`). Where allowed models:
  * `ViT-B/32`
  * `RN50`
  * `RN101`
  * `RN50x4`
  * `RN50x16`
  * `ViT-B/16`
  * `RN50x64`
  * `ViT-L/14`
  * `ViT-L/14@336px`
* --image_folders: List of pairs containing class names and corresponding folder paths containing images. Each pair should be separated by a space, and pairs should be space-separated as well (e.g., "dogs /path/to/dog_images cats /path/to/cat_images").
* --device (optional): Device to run the model on (default: "cpu"). Set to "cuda" if you have a compatible GPU.

Example usage:

```bash
python clip2class_dist/compute.py --clip_model ViT-B/32 --image_folders dogs /path/to/dog_images cats /path/to/cat_images --device cpu
```

## Output

The script will output the following information:

1. Cluster centers for each class.
2. Cluster radii for each class.
3. Dot product distributions within each class.
4. Average dot product between different classes.
5. Minimum dot product between different classes.
6. Maximum dot product between different classes.

Example output:

```bash
Cluster centers: {'dogs': array([...]), 'cats': array([...])}
Cluster radii: {'dogs': 0.2384927, 'cats': 0.27184734}
Dot product distributions: {'dogs': array([...]), 'cats': array([...])}
Average dot product between classes: 0.19983745
Minimum dot product between classes: 0.01759241
Maximum dot product between classes: 0.38273957
```

## License

This project is released under the MIT License.
