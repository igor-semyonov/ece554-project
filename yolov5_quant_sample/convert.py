import pickle as pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt


def main():
    data_dir = Path.home() / "Downloads" / "cifar-10-batches-py"
    with open(data_dir / "batches.meta", "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    global names
    names = meta["label_names"]

    for batch_idx in range(1, 6):
        batch_file = data_dir / f"data_batch_{batch_idx}"
        images, labels = read_batch(batch_file)
        try:
            train_images = np.concatenate([train_images, images])
            train_labels += labels
        except NameError:
            train_images = images
            train_labels = labels
    test_images, test_labels = read_batch(data_dir / "test_batch")

    #  display an image
    #  img: np.ndarray = train_images[-1, :, :, :]
    #  img = img.reshape((32, 32, 3))
    #  plt.imshow(img)
    #  plt.get_current_fig_manager().window.showMaximized()
    #  plt.show()
    #  return

    destination_dir = Path("./cifar10")
    shutil.rmtree(destination_dir, ignore_errors=True)
    export_task(
        "train",
        destination_dir,
        train_images,
        train_labels,
    )
    export_task(
        "val",
        destination_dir,
        test_images,
        test_labels,
    )


def read_batch(
    batch_file: Path,
) -> (np.ndarray[np.uint8], list[int]):
    with open(batch_file, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    labels = batch["labels"]
    labels = [f"{c} 0.5 0.5 1.0 1.0" for c in labels]
    batch_size = len(labels)
    images = np.transpose(batch["data"].reshape((batch_size, 3, 32, 32)), (0, 3, 2, 1))
    return images, labels


def export_task(
    task: str,
    destination_dir: Path,
    images: np.ndarray[np.uint8],
    labels: list[int],
):
    image_dir = destination_dir / "images" / task
    label_dir = destination_dir / "labels" / task
    for dir in (image_dir, label_dir):
        dir.mkdir(exist_ok=True, parents=True)

    dict_yaml = {
        "train": str(destination_dir / "train.txt"),
        "val": str(destination_dir / "val.txt"),
        "names": names,
        "nc": len(names),
    }
    with open(destination_dir / "data.yaml", "w") as f:
        yaml.dump(dict_yaml, f)

    for idx, (image, label) in enumerate(zip(images, labels)):
        sample_name = str(idx).zfill(6)
        image_file = image_dir / (sample_name + ".png")
        cv2.imwrite(str(image_file), image)
        (label_dir / (sample_name + ".txt")).write_text(label)
        with open(destination_dir / f"{task}.txt", "a") as f:
            f.write(f"{str(image_file)}\n")


if __name__ == "__main__":
    main()
