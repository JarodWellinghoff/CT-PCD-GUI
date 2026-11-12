package image

type ImageOutputConfig struct {
	RescaleToHU bool    `json:"RescaleToHU" yaml:"RescaleToHU"`
	AngleOffset float64 `json:"AngleOffset" yaml:"AngleOffset"` // counterclockwise rotation; offsets the reconstruction grid *during* backprojection (i.e. no spatial resolution hit b/c of rotation)
	Binary      string  `json:"Binary" yaml:"Binary"`
	DICOM       string  `json:"DICOM" yaml:"DICOM"`
	NIfTI1      string  `json:"NIfTI1" yaml:"NIfTI1"`
	Png         string  `json:"Png" yaml:"Png"`
}
