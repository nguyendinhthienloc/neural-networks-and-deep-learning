"""Export a few MNIST images and run them through network2.

Outputs:
  test_input/*.png              viewable grayscale image files
  test_input/*_pixels.txt       original 28x28 numeric pixel grids
  test_input/*_from_png.txt     numbers decoded back from the PNG file
  test_output/predictions.csv   before/after predictions
  test_output/run_report.md     short pipeline documentation
"""

import csv
import os
import struct
import random
import zlib

import numpy as np

import mnist_loader
import network2


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT_DIR, "test_input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "test_output")


def log_step(message):
    print("[run_test_images] {}".format(message))


def ensure_dirs():
    log_step("ensure_dirs: create test_input and test_output if needed")
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def image_matrix(x):
    return x.reshape(28, 28)


def png_chunk(chunk_type, data):
    chunk = chunk_type + data
    return (
        struct.pack(">I", len(data)) +
        chunk +
        struct.pack(">I", zlib.crc32(chunk) & 0xffffffff))


def write_png(path, pixels):
    """Write a dependency-free 8-bit grayscale PNG."""
    log_step("write_png: {}".format(os.path.relpath(path, ROOT_DIR)))
    values = np.clip(pixels * 255, 0, 255).astype(np.uint8)
    height, width = values.shape
    raw_rows = b"".join(b"\x00" + values[row].tobytes() for row in range(height))
    png = b"".join([
        b"\x89PNG\r\n\x1a\n",
        png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
        png_chunk(b"IDAT", zlib.compress(raw_rows)),
        png_chunk(b"IEND", b""),
    ])
    with open(path, "wb") as f:
        f.write(png)


def paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_numbers(path):
    """Read the grayscale PNGs produced by write_png back into 0..1 numbers."""
    log_step("read_png_numbers: {}".format(os.path.relpath(path, ROOT_DIR)))
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file: {}".format(path))

    pos = 8
    width = height = color_type = bit_depth = None
    compressed = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        chunk_data = data[pos+8:pos+8+length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk_data)
            if bit_depth != 8 or color_type != 0 or interlace != 0:
                raise ValueError("Only 8-bit non-interlaced grayscale PNG is supported")
        elif chunk_type == b"IDAT":
            compressed.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    raw = zlib.decompress(b"".join(compressed))
    rows = []
    previous = [0] * width
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        encoded = list(raw[offset:offset+width])
        offset += width
        row = []
        for col, value in enumerate(encoded):
            left = row[col-1] if col else 0
            up = previous[col]
            upper_left = previous[col-1] if col else 0
            if filter_type == 0:
                decoded = value
            elif filter_type == 1:
                decoded = value + left
            elif filter_type == 2:
                decoded = value + up
            elif filter_type == 3:
                decoded = value + ((left + up) // 2)
            elif filter_type == 4:
                decoded = value + paeth_predictor(left, up, upper_left)
            else:
                raise ValueError("Unsupported PNG filter: {}".format(filter_type))
            row.append(decoded & 0xff)
        rows.append(row)
        previous = row
    return np.array(rows, dtype=float) / 255.0


def write_pixel_grid(path, pixels):
    log_step("write_pixel_grid: {}".format(os.path.relpath(path, ROOT_DIR)))
    with open(path, "w") as f:
        for row in pixels:
            f.write(" ".join("{:.2f}".format(v) for v in row))
            f.write("\n")


def output_summary(net, x):
    log_step("output_summary: feed image through network")
    output = net.feedforward(x)
    ranked = np.argsort(output.ravel())[::-1]
    return {
        "prediction": int(ranked[0]),
        "confidence": float(output[ranked[0]][0]),
        "top3": [(int(i), float(output[i][0])) for i in ranked[:3]],
        "raw": output.ravel(),
    }


def export_samples(samples):
    log_step("export_samples: write PNGs and pixel grids")
    exported = []
    for index, (x, label) in enumerate(samples):
        pixels = image_matrix(x)
        stem = "sample_{:02d}_label_{}".format(index, label)
        png_path = os.path.join(INPUT_DIR, stem + ".png")
        grid_path = os.path.join(INPUT_DIR, stem + "_pixels.txt")
        png_grid_path = os.path.join(INPUT_DIR, stem + "_from_png.txt")
        write_png(png_path, pixels)
        write_pixel_grid(grid_path, pixels)
        write_pixel_grid(png_grid_path, read_png_numbers(png_path))
        exported.append((stem, png_path, grid_path, png_grid_path))
    return exported


def write_predictions(rows):
    path = os.path.join(OUTPUT_DIR, "predictions.csv")
    log_step("write_predictions: {}".format(os.path.relpath(path, ROOT_DIR)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "true_label",
                "before_prediction",
                "before_confidence",
                "after_prediction",
                "after_confidence",
                "after_top3",
            ])
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_report(exported, prediction_rows, training_cost, training_accuracy):
    path = os.path.join(OUTPUT_DIR, "run_report.md")
    log_step("write_report: {}".format(os.path.relpath(path, ROOT_DIR)))
    with open(path, "w") as f:
        f.write("# MNIST Test Image Pipeline\n\n")
        f.write("This run exports a few MNIST test images, trains a small ")
        f.write("`network2.Network([784, 30, 10])`, and records predictions.\n\n")

        f.write("## Pipeline\n\n")
        f.write("1. `mnist_loader.load_data_wrapper()` reads `data/mnist.pkl.gz`.\n")
        f.write("2. This script exports selected test samples as real `.png` files.\n")
        f.write("3. The PNG files are decoded back into 28x28 brightness numbers.\n")
        f.write("4. Each 28x28 grid is reshaped into a `(784, 1)` column vector.\n")
        f.write("5. Each pixel is a brightness number from `0.0` to `1.0`.\n")
        f.write("6. Training labels become one-hot vectors, such as digit 5 -> ")
        f.write("`[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`.\n")
        f.write("7. `feedforward()` calculates `sigmoid(dot(weights, input) + bias)` ")
        f.write("layer by layer.\n")
        f.write("8. `backprop()` calculates how weights and biases should change.\n")
        f.write("9. `update_mini_batch()` applies those changes using the learning rate.\n\n")

        f.write("## Training Run\n\n")
        f.write("- Architecture: `784 -> 30 -> 10`\n")
        f.write("- Training examples: first `1000` MNIST training images\n")
        f.write("- Epochs: `3`\n")
        f.write("- Mini-batch size: `10`\n")
        f.write("- Learning rate eta: `3.0`\n")
        f.write("- Training costs: `{}`\n".format(
            [round(float(c), 4) for c in training_cost]))
        f.write("- Training accuracy: `{}`\n\n".format(training_accuracy))

        f.write("## Exported Inputs\n\n")
        for stem, png_path, grid_path, png_grid_path in exported:
            f.write("- `{}`: `{}`, `{}`, `{}`\n".format(
                stem,
                os.path.relpath(png_path, ROOT_DIR),
                os.path.relpath(grid_path, ROOT_DIR),
                os.path.relpath(png_grid_path, ROOT_DIR)))

        f.write("\n## Predictions\n\n")
        f.write("| sample | true | before | after | after top 3 |\n")
        f.write("| --- | ---: | ---: | ---: | --- |\n")
        for row in prediction_rows:
            f.write("| {sample} | {true_label} | {before_prediction} | "
                    "{after_prediction} | {after_top3} |\n".format(**row))
    return path


def main():
    log_step("main: seed random number generators")
    random.seed(11)
    np.random.seed(11)
    log_step("main: prepare output folders")
    ensure_dirs()

    log_step("main: load MNIST data")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    log_step("main: select training subset and sample images")
    training_subset = training_data[:1000]
    samples = test_data[:8]

    log_step("main: create network")
    net = network2.Network([784, 30, 10], pipeline_logging=True)
    log_step("main: record predictions before training")
    before = [output_summary(net, x) for x, _ in samples]

    log_step("main: train network")
    training_cost, training_accuracy = net.SGD(
        training_subset,
        epochs=3,
        mini_batch_size=10,
        eta=3.0,
        monitor_training_cost=True,
        monitor_training_accuracy=True)[2:]

    log_step("main: record predictions after training")
    after = [output_summary(net, x) for x, _ in samples]
    log_step("main: export sample images")
    exported = export_samples(samples)

    log_step("main: assemble prediction rows")
    prediction_rows = []
    for (stem, _, _, _), (_, label), before_result, after_result in zip(
            exported, samples, before, after):
        prediction_rows.append({
            "sample": stem,
            "true_label": int(label),
            "before_prediction": before_result["prediction"],
            "before_confidence": "{:.4f}".format(before_result["confidence"]),
            "after_prediction": after_result["prediction"],
            "after_confidence": "{:.4f}".format(after_result["confidence"]),
            "after_top3": "; ".join(
                "{}:{:.3f}".format(digit, score)
                for digit, score in after_result["top3"]),
        })

    predictions_path = write_predictions(prediction_rows)
    report_path = write_report(
        exported, prediction_rows, training_cost, training_accuracy)

    print("Wrote {}".format(os.path.relpath(predictions_path, ROOT_DIR)))
    print("Wrote {}".format(os.path.relpath(report_path, ROOT_DIR)))


if __name__ == "__main__":
    main()
