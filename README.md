# Binarization algorithms

The Python script ```src/binarization.py``` allows you to binarize an image using various global and local thresholding methods, such as Otsu, Unbalanced Otsu, Niblack, and Multiscale Niblack. The binarization process involves converting the image into a black and white format by assigning a threshold value that separates the foreground (objects of interest) from the background.

## Options:

* `--image-path` or `-i` (required): the path to the input image file or folder
* `--method` or `-m` (optional, default: otsu): the method to use for binarization. The available methods are "otsu", "unbalanced_otsu", "niblack", and "multiscale_niblack".
* `--window-scope` or `-w` (optional, default: 5): the window scope parameter used in Niblack and Multiscale Niblack methods. Window size is window_scope*2 + 1
* `-k` (optional, default: -0.2): the k parameter used in Niblack method. Threshold calculated as mean + k * std + a.
* `-a` (optional, default: 0): the a parameter used in Niblack method. Threshold calculated as mean + k * std + a.
* `--std-threshold` or `-s` (optional, default: 5): the std threshold used in Multiscale Niblack method. If std < std_threshold, window size doubles.
* `--output-path` or `-o` (required): the path to the output image file or directory. If the --image-path parameter is a directory, then this parameter must be a directory. 
* `--allow-convert` or `-c` (optional): if specified, allows conversion of the input image to grayscale before binarization. If not specifield, then input image must be grayscale with 1 channel
* `--invert-result` (optional): if specified, inverts the resulting mask. By default background is 255 and foreground is 0.

The script uses the click library to parse command-line arguments and provides a command-line interface for the user. The binarize function is the entry point to the script and performs the binarization of the image. The function first checks whether the input path is a file or a directory. If it is a file, it calls the binarize_and_save_image function to binarize the image and save the output to the specified output path. If it is a directory, it retrieves a list of image files from the directory and passes them to the binarize_and_save_image function in parallel using the multiprocessing.Pool library. The tqdm library is used to display a progress bar for the parallel execution.

The binarize_and_save_image function performs the actual binarization using the selected method and saves the output to the specified output path. If the --allow-convert parameter is specified, it converts the image to grayscale using the PIL.Image.convert("L") before binarization. If the --invert-result parameter is specified, it inverts the resulting mask using numpy.invert function.

To use the script, simply run it from the command line and specify the required parameters.

```
python src/binarization.py [OPTIONS]
```

## Usage:

Example 1: Binarize a single image using Otsu's method
```
python src/binarization.py --image-path input_image.jpg --method otsu --output-path output_image.png
```

Example 2: Binarize a directory of images using Niblack's method with a window scope of 10 and a k value of -0.1
```
python src/binarization.py --image-path input_images/ --method niblack --window-scope 10 --k -0.1 --output-path output_images/ --allow-convert
```

Example 3: Binarize a single image using Multiscale Niblack's method with a standard deviation threshold of 10 and invert the resulting image
```
python src/binarization.py --image-path input_image.jpg --method multiscale_niblack --std-threshold 10 --output-path output_image.png --invert-result
```

Example 4: Binarize a directory of images using Unbalanced Otsu's method
```
python src/binarization.py --image-path input_images/ --method unbalanced_otsu --output-path output_images/ --allow-convert
``` 

# Metrics

The Python script ```src/metrics.py``` calculates image metrics such as precision, recall, and F-measure between a binary image and a corresponding binary mask. The script uses click library to parse command-line arguments and provides a command-line interface for the user.

## Options

- `--image-path` or `-i` (required): the path to the binary image file or directory containing binary images.
- `--mask-path` or `-m` (required): the path to the binary mask file or directory containing binary masks.
- `--metrics` (optional, default: `["precision", "recall", "f_measure"]`): the list of metrics to calculate. Available options are `precision`, `recall`, and `f_measure`.
- `--allow-convert` or `-c` (optional): if specified, allows conversion of the input image to grayscale before binarization.
- `--invert-mask` (optional): if specified, inverts the mask before calculating metrics.


To use the script, simply run it from the command line and specify the required parameters.

```bash
python image_metrics.py --image-path <binary-image-path> --mask-path <binary-mask-path> [--metrics <metric1> <metric2> ...] [--allow-convert] [--invert-mask]
```

## Usage

Example 1: To calculate precision, recall, and F-measure between a binary image and a binary mask, run the following command:

```
python image_metrics.py --image-path /path/to/binary/image.png --mask-path /path/to/binary/mask.png
```

Example 2: To calculate the same metrics for multiple images in a directory, run the following command:

```
python image_metrics.py --image-path /path/to/images/directory/ --mask-path /path/to/masks/directory/
```

Example 3: To calculate only precision and recall metrics for a single image and convert it to grayscale before binarization, run the following command:

```
python image_metrics.py --image-path /path/to/binary/image.png --mask-path /path/to/binary/mask.png --metrics precision recall --allow-convert
```
