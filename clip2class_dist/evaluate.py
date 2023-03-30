import os
import argparse
from typing import Dict, List, Tuple
import numpy as np
import torch
import clip
import faiss
from PIL import Image


def get_embeddings(model: torch.nn.Module, device: str, image_paths: List[str]) -> np.ndarray:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    embeddings = []

    for image in images:
        image_tensor = clip.tokenize([image]).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embeddings.append(embedding.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def compute_cluster_centers(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    cluster_centers = {}
    for class_name, embed in embeddings.items():
        center = np.mean(embed, axis=0)
        cluster_centers[class_name] = center
    return cluster_centers


def compute_cluster_radii(
    cluster_centers: Dict[str, np.ndarray], embeddings: Dict[str, np.ndarray]
) -> Dict[str, float]:
    cluster_radii = {}
    for class_name, center in cluster_centers.items():
        cluster_radii[class_name] = np.max(np.linalg.norm(embeddings[class_name] - center, axis=1))
    return cluster_radii


def compute_dot_product_distributions(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    dot_product_distributions = {}
    for class_name, embed in embeddings.items():
        # Create a faiss index
        index = faiss.IndexFlatIP(embed.shape[1])
        index.add(embed)
        
        # Compute the inner product between the embeddings of the same class
        k = embed.shape[0]  # The number of nearest neighbors to search for
        D, _ = index.search(embed, k)
        
        # Set the diagonal elements to zero
        np.fill_diagonal(D, 0)
        
        # Store the upper triangular part of the dot product matrix
        dot_product_distributions[class_name] = D[np.triu_indices(D.shape[0], k=1)]
    return dot_product_distributions


def compute_cross_class_dot_products(embeddings: Dict[str, np.ndarray]) -> List[float]:
    all_dot_products = []
    for class_a, emb_a in embeddings.items():
        for class_b, emb_b in embeddings.items():
            if class_a == class_b:
                continue
            dot_products = faiss.inner_product(emb_a, emb_b)
            all_dot_products.extend(dot_products.flatten().tolist())
    return all_dot_products


def main(args: argparse.Namespace):
    # Load CLIP model
    model, _ = clip.load(args.clip_model, device=args.device)
    model.eval()

    # Extract embeddings
    class_embeddings = {}
    for class_name, folder_path in args.image_folders:
        image_paths = [
            os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".jpg", ".jpeg", ".png"))
        ]
        class_embeddings[class_name] = get_embeddings(model, args.device, image_paths)

    # Compute cluster centers and radii
    cluster_centers = compute_cluster_centers(class_embeddings)
    cluster_radii = compute_cluster_radii(cluster_centers, class_embeddings)

    # Compute dot products within each class
    dot_product_distributions = compute_dot_product_distributions(class_embeddings)

    # Compute average, min, and max dot products between classes
    cross_class_dot_products = compute_cross_class_dot_products(class_embeddings)
    avg_dot_product = np.mean(cross_class_dot_products)
    min_dot_product = np.min(cross_class_dot_products)
    max_dot_product = np.max(cross_class_dot_products)

    print("Cluster centers:", cluster_centers)
    print("Cluster radii:", cluster_radii)
    print("Dot product distributions:", dot_product_distributions)
    print("Average dot product between classes:", avg_dot_product)
    print("Minimum dot product between classes:", min_dot_product)
    print("Maximum dot product between classes:", max_dot_product)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model", type=str, required=True, help="Name of the CLIP model.")
    parser.add_argument(
        "--image_folders",
        nargs="+",
        metavar=("class", "folder"),
        action="append",
        required=True,
        help="List of class-folder pairs.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (e.g., 'cpu' or 'cuda').")
    args = parser.parse_args()

    main(args)
