package image

type ImageReader interface {
	Read() ([]*ReconstructedImage, error)
}
