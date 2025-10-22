//! Implementations for ndarray casting and conversions for the ImageBuffer

use image::{ImageBuffer, Pixel};
use ndarray::{Array3, ArrayView3, ArrayViewMut, ArrayViewMut3};
use crate::error::{Error, Result};


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
    fn as_mut_ndarray<'a>(&'a mut self) -> ArrayViewMut3<'a, ImageContainer>;

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

    fn to_ndarray(self) -> Array3<C>{
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

        let vec_data = unsafe {
             Vec::from_raw_parts(data, size, size)
        };
        Self::from_raw(width as u32, height as u32, vec_data).ok_or(Error::ImageConstructFailed)
    }

    fn as_mut_ndarray<'a>(&'a mut self) -> ArrayViewMut3<'a, C> {
        let (width, height) = self.dimensions();
        
        unsafe {
            ArrayViewMut::from_shape_ptr(
                (height as usize, width as usize, P::CHANNEL_COUNT as usize),
                self.as_mut_ptr(),
            )
        }
    }
}


#[cfg(test)]
mod test{
    use image::{Rgb32FImage, Rgba32FImage};
    use super::*;

    #[test]
    fn test_as_ndarray() {
        let (width, height, channels) = (256, 128, 4);
        let data = create_test_data(width, height, channels);
        let test_image = Rgba32FImage::from_vec(256, 128, data).unwrap();

        let array  = test_image.as_ndarray();

        for ((y, x, channel),  value) in array.indexed_iter() {
            assert_eq!(test_image.get_pixel(x as u32, y as u32)[channel], *value);
        }
    }


    #[test]
    fn test_as_mut_ndarray() {
        let (width, height, channels) = (256, 128, 4);
        let data = create_test_data(width, height, channels);
        let mut test_image = Rgba32FImage::from_vec(256, 128, data).unwrap();
        let compare = test_image.clone();

        
        let mut array  = test_image.as_mut_ndarray();
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
        
        let mut array  = test_image.clone().to_ndarray();

        array += 1.0;
        for ((y, x, channel),  value) in array.indexed_iter() {
            assert_eq!(test_image.get_pixel(x as u32, y as u32)[channel] + 1.0, *value);
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
            for (channel, value) in pixel.channels().iter().enumerate(){
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
        let test_image = Array3::from_shape_vec((height as usize, width as usize, channels), data).unwrap();

        let result = Rgb32FImage::from_ndarray(test_image).err().unwrap();
        
        assert_eq!(result, Error::ChannelMismatch);
    }
}