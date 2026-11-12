package image

type ImageWriter interface {
	String() string // return descriptive string about current image writer
	Write(ImageReader) error
}
