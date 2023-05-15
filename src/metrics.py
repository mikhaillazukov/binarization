import os
from collections import defaultdict

import click
import numpy as np
from PIL import Image
from tqdm import tqdm


def precision_score(binarized_image, mask, invert=False):
    fg = 255 if invert else 0
    bg = 0 if invert else 255
    true_positives = np.logical_and(binarized_image == fg, mask == fg).sum()
    false_positives = np.logical_and(binarized_image == fg, mask == bg).sum()
    precision = true_positives / (true_positives + false_positives)

    return precision


def recall_score(binarized_image, mask, invert=False):
    fg = 255 if invert else 0
    bg = 0 if invert else 255
    true_positives = np.logical_and(binarized_image == fg, mask == fg).sum()
    false_negatives = np.logical_and(binarized_image == bg, mask == fg).sum()
    recall = true_positives / (true_positives + false_negatives)

    return recall


def f_measure_score(binarized_image, mask, invert=False):
    precision = precision_score(binarized_image, mask, invert)
    recall = recall_score(binarized_image, mask, invert)
    f_measure = 2 * (precision * recall) / (precision + recall)

    return f_measure


METRICS_DICT = {
    "precision": precision_score,
    "recall": recall_score,
    "f_measure": f_measure_score
}


def read_image_and_mask(image_path, mask_path, allow_convert):
    if not os.path.exists(image_path):
        raise click.BadParameter(f"{image_path} doesn't exists")
    if not os.path.exists(mask_path):
        raise click.BadParameter(f"{mask_path} doesn't exists")
    binarized_image = Image.open(image_path)
    binarized_mask = Image.open(mask_path)

    if binarized_mask.mode != "L" or binarized_image.mode != "L":
        if not allow_convert:
            raise click.BadParameter(f"Image mode {binarized_image.mode}, mask mode {binarized_mask.mode} Image and mask must be grayscale")
        else:
            binarized_image = binarized_image.convert("L")
            binarized_mask = binarized_mask.convert("L")

    binarized_image = np.array(binarized_image, dtype=np.uint8)
    binarized_mask = np.array(binarized_mask, dtype=np.uint8)

    return binarized_image, binarized_mask


def calculate_image_metrics(binarized_image, binarized_mask, metrics, invert):
    return {metric_name: METRICS_DICT[metric_name](
        binarized_image, binarized_mask, invert) for metric_name in metrics}


@click.command()
@click.option("--image-path", "-i",
              type=click.Path(exists=True),
              required=True,
              help="Binary image path or image dir path")
@click.option("--mask-path", "-m",
              type=click.Path(exists=True),
              required=True,
              help="Mask path or mask dir path")
@click.option("--metrics",
              type=click.Choice(
                  ["precision", "recall", "f_measure"],
                  case_sensitive=False),
              multiple=True,
              default=["precision", "recall", "f_measure"])
@click.option("--allow-convert", "-c",
              is_flag=True,
              default=False,
              help="Allow to convert image to grayscale")
@click.option("--invert",
              is_flag=True,
              default=False,
              help="Calculate for invert photometric")
def compute_metrics(image_path, mask_path, metrics, allow_convert, invert):
    if os.path.isfile(image_path) and os.path.isfile(mask_path):
        binarized_image, binarized_mask = read_image_and_mask(image_path,
                                                              mask_path,
                                                              allow_convert)
        metrics_result = calculate_image_metrics(
            binarized_image, binarized_mask, metrics, invert)
    elif os.path.isdir(image_path) and os.path.isdir(mask_path):
        image_names = sorted(os.listdir(image_path))
        metrics_result = defaultdict(lambda: np.zeros(
            len(image_names), dtype=np.float32))
        for i, image_name in enumerate(tqdm(image_names, desc="Calculating metrics")):
            binarized_image, binarized_mask = read_image_and_mask(
                os.path.join(image_path, image_name),
                os.path.join(mask_path, image_name),
                allow_convert)

            for metric_name, result in calculate_image_metrics(binarized_image, binarized_mask, metrics, invert).items():
                metrics_result[metric_name][i] = result
        metrics_result = {metric_name: np.mean(
            result) for metric_name, result in metrics_result.items()}
    else:
        raise click.BadParameter("Image path or mask path is invalid")
    click.echo(
        "\n".join(
            [f"{metric_name}: {result*100:.2f}" for metric_name,
             result in metrics_result.items()]
        )
    )


if __name__ == "__main__":
    compute_metrics()
