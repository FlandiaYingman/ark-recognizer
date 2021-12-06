# Ark Recognizer

A recognizer to recognize items of *Arknights* screenshots.

## Principle

This project uses *OpenCV* to deal with image processing.

### Preprocess: Load Items

Firstly, the recognizer loads the icons and data of the items. Read [Update `Items`](items/update_items.md) to obtain
more information about items icons and data.

It simply loads the ID, the name, and the icon image of each item.

Then, the recognizer is ready to recognize any *Arknights* screenshots.

### First Round: Feature Matching

In this round, the recognizer detects the keypoints and descriptor (i.e. features) of all items and the screenshot. For
each item, the recognizer matches its features with the screenshot's. If there are good matches, the recognizer computes
the perspective transformation from the item to screenshot, otherwise, the recognizer assumes the item is absent.
Therefore, the recognizer will know where's the item is in the screenshot.

Currently, the feature detection algorithm is *SIFT*, and the matcher algorithm is *Brute-Force*.

