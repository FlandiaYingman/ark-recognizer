# Ark Recognizer

Recognizes the type and number of items from *Arknights* screenshots.

## Usage

Run `src/ark_recognizer_py/recognizer.py` with some screenshots file as arguments. It will print out
a [ArkPlanner](https://penguin-stats.io/planner) compatible json as output. (More output formats coming soon...)

## Run Tests

Run the unit tests under `tests`. If everything goes right, the unit tests shall return a success result.

## Principle

Ark Recognizer loads multiple input scene images (screenshots). To complete the recognition, it does the following steps
sequentially.

### *Circle Detection (Hough Circles)*: to detect the location and size of items in the scene.

In this step, Ark Recognizer recognizes the location and the size of items in the scene.

Consider the table below. It's easy to find out the formula from scene size to icon size. If the aspect ratio is less
than 16:9, the icon diameter will
be: <img src="https://render.githubusercontent.com/render/math?math=d = \frac{1}{10}(w)">, otherwise, the icon diameter
will remain <img src="https://render.githubusercontent.com/render/math?math=d = \frac{1}{10}(\frac{16}{9}h)">. Where
<img src="https://render.githubusercontent.com/render/math?math=w, h, d"> is the scene width, scene height, and icon
diameter.

| Width (Ratio) | Height (Ratio) | Aspect Ratio | Width   (Pixel) | Height (Pixel) | Icon Diameter |
|---------------|----------------|--------------|-----------------|----------------|---------------|
| 1             | 1              | 1.00         | 1080            | 1080           | N/A           |
| 5             | 4              | 1.25         | 1350            | 1080           | 135           |
| 4             | 3              | 1.33         | 1440            | 1080           | 144           |
| 3             | 2              | 1.50         | 1620            | 1080           | 162           |
| 16            | 10             | 1.60         | 1728            | 1080           | 172           |
| 16            | 9              | 1.78         | 1920            | 1080           | 192           |
| 2             | 1              | 2.00         | 2160            | 1080           | 191           |
| 21            | 9              | 2.33         | 2520            | 1080           | 191           |

Apply *Hough Circles* with the estimated icon diameter. Now the location and the size are found out.

### *Template Matching (Correlation Coefficient Normalized)* :to detect the type of the items

In this step, Ark Recognizer recognizes the type of items in the scene.

The scene image is cropped into slices, with the locations and sizes found in the previous step. Then the scene slices
are scaled same to the item templates.

Each slice is matched with all item templates. The most similar item template is chosen to be the type of this slice. If
the similarity is less than a threshold, the slice will be marked not an item.

### *Hash Distance (Average Hash, Hamming Distance)*: to detect the number of the items

In this step, Ark Recognizer recognizes the number of items in the scene.

Crops the scene slices, so they only contain the number region. After adjusting brightness and contrast, and applying a
threshold filter, each digit in the number region can be extracted by finding contours.

The digit image is hashed and compared to the pre-calculated 0-9 digit hash values. The digits which have the minimum
distance is chosen. They are concatenated into the final number.