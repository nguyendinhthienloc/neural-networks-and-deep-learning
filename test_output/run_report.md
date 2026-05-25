# MNIST Test Image Pipeline

This run exports a few MNIST test images, trains a small `network2.Network([784, 30, 10])`, and records predictions.

## Pipeline

1. `mnist_loader.load_data_wrapper()` reads `data/mnist.pkl.gz`.
2. This script exports selected test samples as real `.png` files.
3. The PNG files are decoded back into 28x28 brightness numbers.
4. Each 28x28 grid is reshaped into a `(784, 1)` column vector.
5. Each pixel is a brightness number from `0.0` to `1.0`.
6. Training labels become one-hot vectors, such as digit 5 -> `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`.
7. `feedforward()` calculates `sigmoid(dot(weights, input) + bias)` layer by layer.
8. `backprop()` calculates how weights and biases should change.
9. `update_mini_batch()` applies those changes using the learning rate.

## Training Run

- Architecture: `784 -> 30 -> 10`
- Training examples: first `1000` MNIST training images
- Epochs: `3`
- Mini-batch size: `10`
- Learning rate eta: `3.0`
- Training costs: `[2.2341, 1.3865, 1.2353]`
- Training accuracy: `[438, 714, 739]`

## Exported Inputs

- `sample_00_label_7`: `test_input\sample_00_label_7.png`, `test_input\sample_00_label_7_pixels.txt`, `test_input\sample_00_label_7_from_png.txt`
- `sample_01_label_2`: `test_input\sample_01_label_2.png`, `test_input\sample_01_label_2_pixels.txt`, `test_input\sample_01_label_2_from_png.txt`
- `sample_02_label_1`: `test_input\sample_02_label_1.png`, `test_input\sample_02_label_1_pixels.txt`, `test_input\sample_02_label_1_from_png.txt`
- `sample_03_label_0`: `test_input\sample_03_label_0.png`, `test_input\sample_03_label_0_pixels.txt`, `test_input\sample_03_label_0_from_png.txt`
- `sample_04_label_4`: `test_input\sample_04_label_4.png`, `test_input\sample_04_label_4_pixels.txt`, `test_input\sample_04_label_4_from_png.txt`
- `sample_05_label_1`: `test_input\sample_05_label_1.png`, `test_input\sample_05_label_1_pixels.txt`, `test_input\sample_05_label_1_from_png.txt`
- `sample_06_label_4`: `test_input\sample_06_label_4.png`, `test_input\sample_06_label_4_pixels.txt`, `test_input\sample_06_label_4_from_png.txt`
- `sample_07_label_9`: `test_input\sample_07_label_9.png`, `test_input\sample_07_label_9_pixels.txt`, `test_input\sample_07_label_9_from_png.txt`

## Predictions

| sample | true | before | after | after top 3 |
| --- | ---: | ---: | ---: | --- |
| sample_00_label_7 | 7 | 7 | 7 | 7:0.744; 9:0.237; 0:0.052 |
| sample_01_label_2 | 2 | 7 | 0 | 0:0.954; 6:0.117; 5:0.048 |
| sample_02_label_1 | 1 | 7 | 1 | 1:0.997; 2:0.007; 8:0.004 |
| sample_03_label_0 | 0 | 7 | 7 | 7:0.522; 0:0.302; 9:0.162 |
| sample_04_label_4 | 4 | 7 | 4 | 4:0.911; 9:0.052; 6:0.036 |
| sample_05_label_1 | 1 | 7 | 1 | 1:0.997; 2:0.009; 8:0.005 |
| sample_06_label_4 | 4 | 7 | 9 | 9:0.649; 4:0.107; 7:0.067 |
| sample_07_label_9 | 9 | 7 | 9 | 9:0.413; 8:0.068; 5:0.060 |
