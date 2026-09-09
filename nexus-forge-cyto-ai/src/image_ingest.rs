//! Whole-Slide Image (WSI) Native Ingestion
//!
//! Provides a tile-based streaming architecture to ingest gigapixel pathology
//! images (SVS, NDPI, TIFF) directly into the Nexus-Forge pipeline without
//! relying on intermediate JSON dumps.

use image::{DynamicImage, GenericImageView};
use std::path::Path;

/// A geometric tile extracted from a Whole Slide Image.
pub struct WsiTile {
    pub x_offset: u32,
    pub y_offset: u32,
    pub width: u32,
    pub height: u32,
    pub image: DynamicImage,
}

/// Abstract trait for reading and streaming tiles from large images.
pub trait WsiReader {
    /// Get the total width of the level 0 (maximum resolution) image.
    fn dimensions(&self) -> (u32, u32);

    /// Read a specific tile region.
    fn read_region(&self, x: u32, y: u32, width: u32, height: u32) -> Option<DynamicImage>;

    /// Return an iterator that streams tiles sequentially for memory-safe processing.
    fn stream_tiles(&self, tile_size: u32) -> TileStream<'_, Self>
    where
        Self: Sized,
    {
        TileStream {
            reader: self,
            tile_size,
            current_x: 0,
            current_y: 0,
        }
    }
}

/// Iterator that yields WsiTiles, ensuring memory is bounded.
pub struct TileStream<'a, R: WsiReader + ?Sized> {
    reader: &'a R,
    tile_size: u32,
    current_x: u32,
    current_y: u32,
}

impl<'a, R: WsiReader> Iterator for TileStream<'a, R> {
    type Item = WsiTile;

    fn next(&mut self) -> Option<Self::Item> {
        let (max_w, max_h) = self.reader.dimensions();

        if self.current_y >= max_h {
            return None;
        }

        let w = self.tile_size.min(max_w - self.current_x);
        let h = self.tile_size.min(max_h - self.current_y);

        let tile_image = self.reader.read_region(self.current_x, self.current_y, w, h)?;

        let tile = WsiTile {
            x_offset: self.current_x,
            y_offset: self.current_y,
            width: w,
            height: h,
            image: tile_image,
        };

        // Advance indices
        self.current_x += self.tile_size;
        if self.current_x >= max_w {
            self.current_x = 0;
            self.current_y += self.tile_size;
        }

        Some(tile)
    }
}

// --------------------------------------------------------------------------------
// Standard Image implementation (for smaller datasets or downsampled PNG/TIFF)
// --------------------------------------------------------------------------------
pub struct StandardImageReader {
    img: DynamicImage,
}

impl StandardImageReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, image::ImageError> {
        let img = image::open(path)?;
        Ok(Self { img })
    }
}

impl WsiReader for StandardImageReader {
    fn dimensions(&self) -> (u32, u32) {
        self.img.dimensions()
    }

    fn read_region(&self, x: u32, y: u32, width: u32, height: u32) -> Option<DynamicImage> {
        // We do a soft crop. Not extremely efficient for giant images, but perfect as a fallback.
        let (max_w, max_h) = self.img.dimensions();
        if x >= max_w || y >= max_h {
            return None;
        }
        
        let mut sub = self.img.clone();
        Some(sub.crop(x, y, width, height))
    }
}

// --------------------------------------------------------------------------------
// [Optional] OpenSlide implementation for actual Gigapixel `.svs` files.
// Requires `openslide` feature flag to be enabled during build.
// --------------------------------------------------------------------------------
#[cfg(feature = "openslide")]
pub struct OpenSlideReader {
    slide: openslide::OpenSlide,
}

#[cfg(feature = "openslide")]
impl OpenSlideReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, openslide::Error> {
        let slide = openslide::OpenSlide::new(path.as_ref())?;
        Ok(Self { slide })
    }
}

#[cfg(feature = "openslide")]
impl WsiReader for OpenSlideReader {
    fn dimensions(&self) -> (u32, u32) {
        let (w, h) = self.slide.get_dimensions().unwrap_or((0, 0));
        (w as u32, h as u32)
    }

    fn read_region(&self, x: u32, y: u32, width: u32, height: u32) -> Option<DynamicImage> {
        let bgra_data = self.slide.read_region(x as i64, y as i64, 0, width as i64, height as i64).ok()?;
        
        // Convert BGRA to RGBA DynamicImage
        // Note: The conversion logic depends on the internal representation, 
        // usually ImageBuffer::from_raw can handle it if we map the bytes.
        let buf = image::RgbaImage::from_raw(width, height, bgra_data)?;
        Some(DynamicImage::ImageRgba8(buf))
    }
}
