//! Implementations for ndarray casting and conversions for the ImageBuffer

use num_traits::{AsPrimitive, ToPrimitive};

#[cfg(feature = "image")]
use crate::error::{Error, Result};
#[cfg(feature = "image")]
use image::{ImageBuffer, Pixel};
#[cfg(feature = "image")]
use ndarray::{Array3, ArrayView3, ArrayViewMut, ArrayViewMut3};

#[cfg(feature = "image")]
/// Conversion methods for working with ndarrays.
///
/// All methods work without copying any data.
pub trait ImageArray<P: image::Pixel, ImageContainer> {
    /// Cast the ImageBuffer as an ArrayView3.
    ///
    /// * `Y` index is the row
    /// * `X` index is the columns
    /// * `Z` index is the channel
    ///
    /// So when referencing:
    /// `array[[y, x, z]]`
    ///
    /// This does not copy the data, as it is a reference to the actual data in the buffer.
    fn as_ndarray<'a>(&'a self) -> ArrayView3<'a, ImageContainer>;

    /// Cast the ImageBuffer as an ArrayViewMut3.
    ///
    /// * `Y` index is the row
    /// * `X` index is the columns
    /// * `Z` index is the channel
    ///
    /// So when referencing:
    /// `array[[y, x, z]]`
    ///
    /// This does not copy the data, as it is a reference to the actual data in the buffer.
    fn as_ndarray_mut<'a>(&'a mut self) -> ArrayViewMut3<'a, ImageContainer>;

    /// Interpret the ImageBuffer as an Array3.
    ///
    /// * `Y` index is the row
    /// * `X` index is the columns
    /// * `Z` index is the channel
    ///
    /// So when referencing:
    /// `array[[y, x, z]]`
    ///
    /// This does not copy the data, but it does consume the buffer.
    fn to_ndarray(self) -> Array3<ImageContainer>;

    /// Convert the provided array into the ImageBuffer
    ///
    /// * `Y` index is the row
    /// * `X` index is the columns
    /// * `Z` index is the channel
    ///
    /// So when referencing:
    /// `array[[y, x, z]]`
    ///
    /// This does not copy the data, but it does consume the buffer.
    fn from_ndarray(array: Array3<ImageContainer>) -> Result<ImageBuffer<P, Vec<ImageContainer>>>;
}

#[cfg(feature = "image")]
impl<P, C> ImageArray<P, C> for ImageBuffer<P, Vec<C>>
where
    P: Pixel<Subpixel = C>,
    C: Clone + Copy,
{
    fn as_ndarray<'a>(&'a self) -> ArrayView3<'a, C> {
        let (width, height) = self.dimensions();
        unsafe {
            ArrayView3::from_shape_ptr(
                (height as usize, width as usize, P::CHANNEL_COUNT as usize),
                self.as_raw().as_ptr(),
            )
        }
    }

    fn to_ndarray(self) -> Array3<C> {
        let (width, height) = self.dimensions();
        unsafe {
            Array3::from_shape_vec_unchecked(
                (height as usize, width as usize, P::CHANNEL_COUNT as usize),
                self.into_raw(),
            )
        }
    }

    fn from_ndarray(mut array: Array3<C>) -> Result<ImageBuffer<P, Vec<C>>> {
        let (height, width, channels) = array.dim();

        if channels != P::CHANNEL_COUNT.into() {
            return Err(Error::ChannelMismatch);
        }

        let data = array.as_mut_ptr();

        std::mem::forget(array);
        let size = height * width * channels;

        let vec_data = unsafe { Vec::from_raw_parts(data, size, size) };
        Self::from_raw(width as u32, height as u32, vec_data).ok_or(Error::ImageConstructFailed)
    }

    fn as_ndarray_mut<'a>(&'a mut self) -> ArrayViewMut3<'a, C> {
        let (width, height) = self.dimensions();

        unsafe {
            ArrayViewMut::from_shape_ptr(
                (height as usize, width as usize, P::CHANNEL_COUNT as usize),
                self.as_mut_ptr(),
            )
        }
    }
}

/// Trait for converting the provided value to a normalized float.
///
/// This is used for image processing where a lot of operations rely on floating values.
pub trait NormalizedFloat<T>
where
    T: AsPrimitive<f32> + AsPrimitive<f64>,
{
    /// Convert the value to a 32 bit float.
    ///
    /// The value will be in a normalized range according to color depths.
    ///
    /// For example in u8, a value of 255 would be represented as 1.0.
    ///
    /// Returns None if it overflows and could not be represented.
    fn to_f32_normalized(&self) -> Option<f32>;
    /// Convert the value to a 64 bit float
    ///
    /// The value will be in a normalized range according to color depths.
    ///
    /// For example in u8, a value of 255 would be represented as 1.0.
    ///
    /// Returns None if it overflows and could not be represented.
    fn to_f64_normalized(&self) -> Option<f64>;

    /// Converts the f32 value to the provided type
    ///
    /// Returns None if it overflows and could not be represented.
    fn from_f32_normalized(value: f32) -> Option<T>;

    /// Converts the f64 value to the provided type
    ///
    /// Returns None if it overflows and could not be represented.
    fn from_f64_normalized(value: f64) -> Option<T>;
}

impl NormalizedFloat<f32> for f32 {
    fn to_f32_normalized(&self) -> Option<f32> {
        Some(*self)
    }

    fn to_f64_normalized(&self) -> Option<f64> {
        self.to_f64()
    }
    fn from_f32_normalized(value: f32) -> Option<f32> {
        Some(value)
    }

    fn from_f64_normalized(value: f64) -> Option<f32> {
        value.to_f32()
    }
}

impl NormalizedFloat<f64> for f64 {
    fn to_f32_normalized(&self) -> Option<f32> {
        self.to_f32()
    }

    fn to_f64_normalized(&self) -> Option<f64> {
        Some(*self)
    }
    fn from_f32_normalized(value: f32) -> Option<f64> {
        value.to_f64()
    }

    fn from_f64_normalized(value: f64) -> Option<f64> {
        Some(value)
    }
}

#[macro_export]
macro_rules! impl_as_float {
    ($type:ty) => {
        impl NormalizedFloat<$type> for $type {
            fn to_f32_normalized(&self) -> Option<f32> {
                self.to_f32()
                    .map(|converted| converted / <$type>::MAX as f32)
            }

            fn to_f64_normalized(&self) -> Option<f64> {
                self.to_f64()
                    .map(|converted| converted / <$type>::MAX as f64)
            }

            fn from_f32_normalized(value: f32) -> Option<$type> {
                Some((value * <$type>::MAX as f32).as_())
            }

            fn from_f64_normalized(value: f64) -> Option<$type> {
                Some((value * <$type>::MAX as f64).as_())
            }
        }
    };
}

impl_as_float!(i32);
impl_as_float!(u32);
impl_as_float!(i16);
impl_as_float!(u16);
impl_as_float!(i8);
impl_as_float!(u8);

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb32FImage, Rgba32FImage};
    use rstest::*;

    #[test]
    fn test_as_ndarray_rgba() {
        let (width, height, channels) = (256, 128, 4);
        let data = create_test_data(width, height, channels);
        let test_image = Rgba32FImage::from_vec(256, 128, data).unwrap();

        let array = test_image.as_ndarray();

        for ((y, x, channel), value) in array.indexed_iter() {
            assert_eq!(test_image.get_pixel(x as u32, y as u32)[channel], *value);
        }
    }

    #[test]
    fn test_as_ndarray_luma() {
        let (width, height, channels) = (256, 128, 1);
        let data = create_test_data(width, height, channels);
        let test_image: ImageBuffer<Luma<f32>, Vec<f32>> =
            ImageBuffer::from_vec(256, 128, data).unwrap();

        let array = test_image.as_ndarray();

        for ((y, x, channel), value) in array.indexed_iter() {
            assert_eq!(test_image.get_pixel(x as u32, y as u32)[channel], *value);
        }
    }

    #[test]
    fn test_as_ndarray_mut() {
        let (width, height, channels) = (256, 128, 4);
        let data = create_test_data(width, height, channels);
        let mut test_image = Rgba32FImage::from_vec(256, 128, data).unwrap();
        let compare = test_image.clone();

        let mut array = test_image.as_ndarray_mut();
        array += 1.0;

        for (x, y, pixel) in test_image.enumerate_pixels() {
            let compare_pixel = compare.get_pixel(x, y);
            for (channel, value) in pixel.channels().iter().enumerate() {
                assert_eq!(*value, compare_pixel[channel] + 1.0);
            }
        }
    }

    #[test]
    fn test_to_ndarray() {
        let (width, height, channels) = (256, 128, 4);
        let data = create_test_data(width, height, channels);
        let test_image = Rgba32FImage::from_vec(256, 128, data).unwrap();

        let mut array = test_image.clone().to_ndarray();

        array += 1.0;
        for ((y, x, channel), value) in array.indexed_iter() {
            assert_eq!(
                test_image.get_pixel(x as u32, y as u32)[channel] + 1.0,
                *value
            );
        }
    }

    #[test]
    fn test_from_ndarray() {
        let (width, height, channels) = (256, 128, 4);
        let data = create_test_data(width, height, channels);
        let test_image = Array3::from_shape_vec((height, width, channels), data).unwrap();
        let compare_data = test_image.clone();

        let result = Rgba32FImage::from_ndarray(test_image).unwrap();

        for (x, y, pixel) in result.enumerate_pixels() {
            for (channel, value) in pixel.channels().iter().enumerate() {
                assert_eq!(*value, compare_data[[y as usize, x as usize, channel]]);
            }
        }
    }

    fn create_test_data(width: usize, height: usize, channels: usize) -> Vec<f32> {
        let total_elements = width * height * channels;
        (0..total_elements).map(|x| (x + 1) as f32).collect()
    }

    #[test]
    fn test_from_ndarray_with_invalid_channels() {
        let channels = 4;
        let (width, height) = (256.0, 128.0);
        let total_elements = (width * height * 4.0) as usize;
        let data: Vec<f32> = (0..total_elements).map(|x| (x + 1) as f32).collect();
        let test_image =
            Array3::from_shape_vec((height as usize, width as usize, channels), data).unwrap();

        let result = Rgb32FImage::from_ndarray(test_image).err().unwrap();

        assert_eq!(result, Error::ChannelMismatch);
    }

    #[rstest]
    #[case(1.0)]
    #[case(255.0)]
    #[case(0.5)]
    #[case(-1.0)]
    #[case(-255.0)]
    fn test_f32(#[case] float: f32) {
        assert_eq!(float.to_f32_normalized().unwrap(), float);
        assert_eq!(f32::from_f32_normalized(float).unwrap(), float);

        let float_64: f64 = float.as_();
        assert_eq!(float_64.to_f64_normalized().unwrap(), float_64);
        assert_eq!(f64::from_f64_normalized(float_64).unwrap(), float_64);

        let converted_to_float64 = float.to_f64_normalized().unwrap();
        assert_eq!(converted_to_float64, float as f64);

        let converted_back_to_float32 = float_64.to_f32_normalized().unwrap();
        assert_eq!(converted_back_to_float32, float);
    }

    #[macro_export]
    macro_rules! test_unsigned_ints {
        ($name:ident, $type:ty) => {
            #[rstest]
            #[case(0)]
            #[case(1)]
            #[case($type::MAX)]
            #[case($type::MIN)]
            fn $name(#[case] int: $type) {
                let normalized_f32 = int.to_f32_normalized().unwrap();
                let expected_normalized_f32 = int as f32 / <$type>::MAX as f32;
                assert_eq!(normalized_f32, expected_normalized_f32);

                let int_from_float32 =
                    <$type>::from_f32_normalized(expected_normalized_f32).unwrap();
                let expected_int_from_float32 =
                    (expected_normalized_f32 * <$type>::MAX as f32) as $type;
                assert_eq!(int_from_float32, expected_int_from_float32);

                let normalized_f64 = int.to_f64_normalized().unwrap();
                let expected_normalized_f64 = int as f64 / <$type>::MAX as f64;
                assert_eq!(normalized_f64, expected_normalized_f64);

                let int_from_float64 =
                    <$type>::from_f64_normalized(expected_normalized_f64).unwrap();
                let expected_int_from_float64 =
                    (expected_normalized_f64 * <$type>::MAX as f64) as $type;
                assert_eq!(int_from_float64, expected_int_from_float64);
            }
        };
    }

    #[macro_export]
    macro_rules! test_signed_ints {
        ($name:ident, $type:ty) => {
            #[rstest]
            #[case(0)]
            #[case(1)]
            #[case($type::MAX)]
            #[case($type::MIN)]
            #[case(-1)]
            #[case(-$type::MAX)]
            fn $name(#[case] int: $type) {
                let normalized_f32 = int.to_f32_normalized().unwrap();
                let expected_normalized_f32 = int as f32 / <$type>::MAX as f32;
                assert_eq!(normalized_f32, expected_normalized_f32);

                let int_from_float32 =
                    <$type>::from_f32_normalized(expected_normalized_f32).unwrap();
                let expected_int_from_float32 =
                    (expected_normalized_f32 * <$type>::MAX as f32) as $type;
                assert_eq!(int_from_float32, expected_int_from_float32);

                let normalized_f64 = int.to_f64_normalized().unwrap();
                let expected_normalized_f64 = int as f64 / <$type>::MAX as f64;
                assert_eq!(normalized_f64, expected_normalized_f64);

                let int_from_float64 =
                    <$type>::from_f64_normalized(expected_normalized_f64).unwrap();
                let expected_int_from_float64 =
                    (expected_normalized_f64 * <$type>::MAX as f64) as $type;
                assert_eq!(int_from_float64, expected_int_from_float64);
            }
        };
    }
    // Using the macro to generate tests for i32
    test_signed_ints!(test_i32, i32);
    test_signed_ints!(test_i16, i16);
    test_signed_ints!(test_i8, i8);
    test_unsigned_ints!(test_u32, u32);
    test_unsigned_ints!(test_u16, u16);
    test_unsigned_ints!(test_u8, u8);
}
