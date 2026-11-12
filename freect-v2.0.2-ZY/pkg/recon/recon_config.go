package recon

import (
	"gitlab.com/freect/freect/pkg/geometry"
	"gitlab.com/freect/freect/pkg/image"
	"gitlab.com/freect/freect/pkg/raw"
)

type ReconConfig struct {
	RawInput      *raw.RawInputConfig      `json:"RawInput" yaml:"RawInput"`
	ScanGeometry  *geometry.ScanGeometry   `json:"ScanGeometry" yaml:"ScanGeometry"`
	Preprocessing *PreprocessingConfig     `json:"Preprocessing" yaml:"Preprocessing"`
	ReconVolume   *ReconVolume             `json:"ReconVolume" yaml:"ReconVolume"`
	ReconSpecific interface{}              `json:"ReconSpecific" yaml:"ReconSpecific"`
	ImageOutput   *image.ImageOutputConfig `json:"ImageOutput" yaml:"ImageOutput"`
}

func NewReconConfig(recon_specific_config interface{}) *ReconConfig {
	return &ReconConfig{
		RawInput:      &raw.RawInputConfig{},
		ScanGeometry:  &geometry.ScanGeometry{},
		Preprocessing: &PreprocessingConfig{},
		ReconVolume:   &ReconVolume{},
		ReconSpecific: recon_specific_config,
		ImageOutput:   &image.ImageOutputConfig{},
	}

}
