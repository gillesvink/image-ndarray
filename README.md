[![Tests](https://github.com/gillesvink/image-ndarray/actions/workflows/test.yaml/badge.svg)](https://github.com/gillesvink/image-ndarray/actions/workflows/test.yaml) 
[![License](https://img.shields.io/crates/l/image-ndarray)](https://crates.io/crates/image-ndarray) 
[![Version](https://img.shields.io/crates/v/image-ndarray)](https://crates.io/crates/image-ndarray) 

# image-ndarray

Zero-copy implementations for the `Image` crate to convert to and from ndarrays.

Working with ndarrays allows for easy manipulation of pixel values.

While there is another crate called `ndarray-image`, that works with dedicated types.
This crate does not and implements the methods directly onto the ImageBuffer objects.

## Usage
Add the crate to your project:
```bash
cargo add image-ndarray
```

Then in your project, when you want to use the ndarrays on the images, make sure to add the prelude:
```rust
use image_ndarray::prelude::*;
```

## Example

Adding a simple value:
```rust
use image::Rgba32FImage;
use image_ndarray::prelude::*;

let mut my_image = Rgba32FImage::new(1920, 1080);

assert!(my_image.to_vec() == vec![0.0; 1920 * 1080 * 4]);

let mut array = my_image.as_mut_ndarray();
array += 1.0;

assert!(my_image.to_vec() == vec![1.0; 1920 * 1080 * 4]);
```

Adding another image:
```rust
use image::Rgba32FImage;
use image_ndarray::prelude::*;

let mut my_image = Rgba32FImage::new(1920, 1080);
let second_image = Rgba32FImage::from_vec(1920, 1080, vec![1.0; 1920 * 1080 * 4]).unwrap();
assert!(my_image.to_vec() == vec![0.0; 1920 * 1080 * 4]);

let mut array = my_image.as_mut_ndarray();
array += &second_image.as_ndarray();

assert!(my_image.to_vec() == vec![1.0; 1920 * 1080 * 4]);
```


Dividing another image (just any math operation supported by [ndarray](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html)):
```rust
use image::Rgba32FImage;
use image_ndarray::prelude::*;

let mut my_image = Rgba32FImage::from_vec(1920, 1080, vec![1.0; 1920 * 1080 * 4]).unwrap();
let second_image = Rgba32FImage::from_vec(1920, 1080, vec![2.0; 1920 * 1080 * 4]).unwrap();
assert!(my_image.to_vec() == vec![1.0; 1920 * 1080 * 4]);

let mut array = my_image.as_mut_ndarray();
array /= &second_image.as_ndarray();

assert!(my_image.to_vec() == vec![0.5; 1920 * 1080 * 4]);
```

Convert image to array:
```rust 
use image::Rgba32FImage;
use image_ndarray::prelude::*;

let my_image = Rgba32FImage::from_vec(1920, 1080, vec![1.0; 1920 * 1080 * 4]).unwrap();
let array = my_image.to_ndarray();

assert!(array.as_slice().unwrap().to_vec() == vec![1.0; 1920 * 1080 * 4]);
```


Or just create an image from the provided array:
```rust
use image::Rgba32FImage;
use image_ndarray::prelude::*;
use ndarray::Array3;

let array = Array3::from_elem((1080, 1920, 4), 1.0);
let my_image = Rgba32FImage::from_ndarray(array).unwrap();

assert!(my_image.to_vec() == vec![1.0; 1920 * 1080 * 4]);
```