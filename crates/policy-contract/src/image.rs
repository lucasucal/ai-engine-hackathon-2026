/// The camera views in `images`, in the order they are listed everywhere.
pub const VIEWS: [&str; 3] = ["wrist", "left", "right"];
pub const FRAME_SIZE: (usize, usize) = (640, 360);
pub const GOAL_SIZE: (usize, usize) = (640, 480);
pub const BLACK_JPEG: &[u8] = include_bytes!("../assets/black.jpg");

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    #[error("not a JPEG")]
    NotJpeg,
    #[error("{width}×{height}, not {}×{}", .expected.0, .expected.1)]
    WrongSize {
        width: usize,
        height: usize,
        expected: (usize, usize),
    },
}

/// Whether `jpeg` is a JPEG of `size`, judged from its header alone.
///
/// # Errors
///
/// How it falls short.
pub fn check_jpeg(jpeg: &[u8], size: (usize, usize)) -> Result<(), FrameError> {
    if !matches!(imagesize::image_type(jpeg), Ok(imagesize::ImageType::Jpeg)) {
        return Err(FrameError::NotJpeg);
    }
    let actual = imagesize::blob_size(jpeg).map_err(|_| FrameError::NotJpeg)?;
    if (actual.width, actual.height) != size {
        return Err(FrameError::WrongSize {
            width: actual.width,
            height: actual.height,
            expected: size,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /// Just enough of a JPEG for its size to be read: start of image, then a baseline frame
    /// header for a 640×480, three-component image.
    const JPEG_640X480: &[u8] = &[
        0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x01, 0xE0, 0x02, 0x80, 0x03, 0x01, 0x22, 0x00,
        0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xD9,
    ];
    const PNG_SIGNATURE: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

    #[test_case(BLACK_JPEG, FRAME_SIZE => Ok(()); "camera frame")]
    #[test_case(JPEG_640X480, GOAL_SIZE => Ok(()); "goal")]
    #[test_case(PNG_SIGNATURE, FRAME_SIZE => Err(FrameError::NotJpeg); "not a JPEG")]
    #[test_case(JPEG_640X480, FRAME_SIZE
        => Err(FrameError::WrongSize { width: 640, height: 480, expected: FRAME_SIZE }); "goal as a frame")]
    #[test_case(BLACK_JPEG, GOAL_SIZE
        => Err(FrameError::WrongSize { width: 640, height: 360, expected: GOAL_SIZE }); "frame as a goal")]
    fn frame_checks(jpeg: &[u8], size: (usize, usize)) -> Result<(), FrameError> {
        check_jpeg(jpeg, size)
    }
}
