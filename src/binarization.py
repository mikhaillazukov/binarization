import multiprocessing
import sys
import os
import istarmap
import click
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def otsu_method(image):
    """Глобальный метод Оцу

    Args:
        image (np.ndarray): Изображение в градациях серого.

    Returns:
        np.ndarray: Бинаризованное изображение
    """
    hist, _ = np.histogram(image, range(0, 257))
    normalized_hist = hist / hist.sum()

    hist_cumsum = np.cumsum(normalized_hist)
    intensity_cumsum = np.cumsum(np.arange(256) * normalized_hist)

    max_variance = -sys.float_info.max
    best_threshold = -1

    for t in range(1, 256):
        w_0 = hist_cumsum[t]
        w_1 = hist_cumsum[-1] - w_0

        if w_0 == 0 or w_1 == 0:
            continue
        mu_0 = intensity_cumsum[t] / w_0
        mu_1 = (intensity_cumsum[-1] - intensity_cumsum[t]) / w_1
        variance = w_0 * w_1 * (mu_0 - mu_1) * (mu_0 - mu_1)

        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    binarized_image = np.zeros(image.shape, dtype=np.uint8)
    binarized_image[image > best_threshold] = 255

    return binarized_image


def unbalanced_otsu_method(image: np.ndarray) -> np.ndarray:
    """Глобальный несбалансированный метод Оцу

    Args:
        image (np.ndarray): Изображение в градациях серого.

    Returns:
        np.ndarray: Бинаризованное изображение
    """
    hist, _ = np.histogram(image, range(0, 257))
    normalized_hist = hist / hist.sum()

    hist_cumsum = np.cumsum(normalized_hist)
    intensity_cumsum = np.cumsum(np.arange(256) * normalized_hist)

    max_variance = -sys.float_info.max
    best_threshold = -1

    for t in range(256):
        w_0 = hist_cumsum[t]
        w_1 = hist_cumsum[-1] - w_0

        if w_0 == 0 or w_1 == 0:
            continue

        g_0 = intensity_cumsum[t]
        g_1 = intensity_cumsum[-1] - intensity_cumsum[t]

        sigma_0 = np.sum(np.square((np.arange(t) - g_0)) * normalized_hist[:t])
        sigma_1 = np.sum(np.square((np.arange(t, 256) - g_1))
                         * normalized_hist[t:])

        sigma_square = w_0 * sigma_0 + w_1 * sigma_1

        variance = w_0*np.log(w_0) + w_1*np.log(w_1) - np.log(sigma_square)
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    binarized_image = np.zeros(image.shape, dtype=np.uint8)
    binarized_image[image > best_threshold] = 255

    return binarized_image


def build_integral_image(image):
    integral_image = np.cumsum(image, axis=0, dtype=np.float64)
    integral_image = np.cumsum(integral_image, axis=1, dtype=np.float64)

    return np.pad(integral_image, (1, 0), constant_values=0)


def niblack_method(image: np.ndarray, window_scope: int = 5,
                   k: float = -0.2, a: float = 0) -> np.ndarray:
    """Реализация локального метода Ниблэка. Вычисление порога происходит по формуле:
    t = mean + k * std + a

    Args:
        image (np.ndarray): Изображение в градациях серого.
        window_scope (int, optional): Размах окна. Далее размер окна вычисляется по формуле window_scope*2+1. Defaults to 5.
        k (float, optional): Настроечный коэффициент. Defaults to -0.2.
        a (float, optional): Настроечный коэффициент. Defaults to 0.
    Returns:
        np.ndarray: Бинаризованное изображение
    """
    window_size = window_scope * 2 + 1
    padded_image = np.pad(image.astype(np.float64),
                          window_scope, mode="reflect")

    integral_image = build_integral_image(padded_image)
    integral_square_image = build_integral_image(np.square(padded_image))

    binarized_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            mean = (integral_image[i + window_size, j + window_size] +
                    integral_image[i, j] -
                    integral_image[i + window_size, j] -
                    integral_image[i, j + window_size])/(window_size * window_size)

            mean_square = (integral_square_image[i + window_size, j + window_size] +
                           integral_square_image[i, j] -
                           integral_square_image[i + window_size, j] -
                           integral_square_image[i, j + window_size])/(window_size * window_size)

            std = np.sqrt(np.clip(mean_square - mean * mean, 0, None))

            t = mean + k * std + a
            binarized_image[i, j] = 255 if image[i, j] >= t else 0

    return binarized_image


def multiscale_niblack_method(image: np.ndarray, window_scope: int = 5,
                              k: float = -0.2, a: float = 0,
                              std_threshold: float = 5):
    """Реализация многомасштабного локального метода Ниблэка.
    Отличительной особенностью данного метода является динамический размер окна,
    который меняется в если дисперсия при текущем размере окна ниже заданного порога.
    Вычисление порога происходит по формуле:
    t = mean + k * std + a

    Args:
        image (np.ndarray): Изображение в градациях серого.
        window_scope (int, optional): Размах окна. Далее размер окна вычисляется по формуле window_scope*2+1. Defaults to 5.
        k (float, optional): Настроечный коэффициент. Defaults to -0.2.
        a (float, optional): Настроечный коэффициент. Defaults to 0.
        std_threshold (float, optional): Пороговое значение дисперсии. При значении ниже порога размер окна будет увеличиваться вдвое
    Returns:
        np.ndarray: Бинаризованное изображение
    """
    pad_width = min(image.shape) // 2
    padded_image = np.pad(image.astype(np.float64), pad_width, mode="reflect")

    integral_image = build_integral_image(padded_image)
    integral_square_image = build_integral_image(np.square(padded_image))

    binarized_image = np.zeros(image.shape, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            std = -1
            window_size = window_scope * 2 + 1

            while std < std_threshold and window_size < min(image.shape):
                mean = (integral_image[i + window_size, j + window_size] +
                        integral_image[i, j] -
                        integral_image[i + window_size, j] -
                        integral_image[i, j + window_size])/(window_size * window_size)

                mean_square = (integral_square_image[i + window_size, j + window_size] +
                               integral_square_image[i, j] -
                               integral_square_image[i + window_size, j] -
                               integral_square_image[i, j + window_size])/(window_size * window_size)

                std = np.sqrt(np.clip(mean_square - mean * mean, 0, None))

                window_size *= 2

            t = mean + k * std + a

            binarized_image[i, j] = 255 if image[i, j] >= t else 0

    return binarized_image


def binarize_and_save_image(image_path: str, method: str, window_scope: int,
                            k: float, a: float, std_threshold: float,
                            output_path: str, allow_convert: bool,
                            invert_result: bool):
    if not os.path.exists(image_path):
        raise click.BadParameter(f"{image_path} doesn't exists")

    image = Image.open(image_path)
    if image.mode != "L":
        if not allow_convert:
            raise click.BadParameter("Image must be grayscale")
        else:
            image = image.convert("L")
    image = np.array(image)
    if method == "otsu":
        binarized_image = otsu_method(image)
    elif method == "unbalanced_otsu":
        binarized_image = unbalanced_otsu_method(image)
    elif method == "niblack":
        binarized_image = niblack_method(image, window_scope, k, a)
    elif method == "multiscale_niblack":
        binarized_image = multiscale_niblack_method(
            image, window_scope, k, a, std_threshold)
    else:
        raise click.BadParameter(f"Unknown method: {method}")
    if invert_result:
        binarized_image = np.invert(binarized_image)
    binarized_image = Image.fromarray(binarized_image)
    binarized_image.save(output_path)


@click.command()
@click.option("--image-path", "-i",
              type=click.Path(exists=True),
              required=True,
              help="Input image path")
@click.option("--method", "-m",
              type=click.Choice(["otsu",
                                 "unbalanced_otsu",
                                 "niblack",
                                 "multiscale_niblack"], case_sensitive=False),
              default="otsu",
              help="Method to use")
@click.option("--window-scope", "-w",
              type=click.INT,
              default=5,
              help="Window scope. Window size is window_scope*2 + 1")
@click.option("-k",
              type=click.FLOAT,
              default=-0.2,
              help="Parameter k for Niblack method. \
              Threshold calculated as mean + k * std + a")
@click.option("-a",
              type=click.FLOAT,
              default=0,
              help="Parameter a for Niblack method. \
              Threshold calculated as mean + k * std + a")
@click.option("--std-threshold", "-s",
              type=click.FLOAT,
              default=5,
              help="Std threshold for Multiscale Niblack method. \
              If std < std_threshold, window size doubles")
@click.option("--output-path", "-o",
              type=click.Path(),
              required=True,
              help="Output image path")
@click.option("--allow-convert", "-c",
              is_flag=True,
              default=False,
              help="Allow converting to grayscale")
@click.option("--invert-result",
              is_flag=True,
              default=False,
              help="Invert mask")
def binarize(image_path: str, method: str, window_scope: int,
             k: float, a: float, std_threshold: float,
             output_path: str, allow_convert: bool,
             invert_result: bool):
    click.echo(f"Method: {method}")
    if os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    else:
        if os.path.dirname(output_path) != "":
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.isfile(image_path):
        if os.path.isdir(output_path):
            output_path = os.path.join(
                output_path, os.path.basename(image_path))

        binarize_and_save_image(image_path, method, window_scope, k, a,
                                std_threshold, output_path, allow_convert,
                                invert_result)
        click.echo(f"Image saved to {output_path}")
    else:
        if os.path.isfile(output_path):
            raise click.BadParameter("Output path must be a directory")
        image_names = sorted(os.listdir(image_path))
        new_data = []
        for image_name in image_names:
            binarized_image_name, _ = os.path.splitext(image_name)
            new_data.append((
                os.path.join(image_path, image_name),
                method, window_scope, k, a,
                std_threshold, os.path.join(
                    output_path, f"{binarized_image_name}.png"),
                allow_convert, invert_result)
            )
        with multiprocessing.Pool() as pool:
            for _ in tqdm(pool.istarmap(binarize_and_save_image, new_data),
                          total=len(new_data)):
                pass


if __name__ == "__main__":
    binarize()
