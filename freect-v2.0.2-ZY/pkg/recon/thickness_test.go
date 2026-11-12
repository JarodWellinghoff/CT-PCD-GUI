package recon

import (
	"image"
	"testing"
)

func TestThickness(t *testing.T) {
	output_volume := &ReconVolume{
		StartPosition:  -10,
		EndPosition:    10.0,
		SliceThickness: 2.0,
		SliceSpacing:   2.0,
		Nx:             4,
		Ny:             4,
	}

	native_slice_thickness := 0.6
	slice_locations := image.GenerateSliceLocations(StartPosition, EndPosition, native_slice_thickness)

	input_stack := make([]*image.ReconstructedImage, len(slice_locations))
	for i, v := range input_stack {
		input_stack := &image.NewReconstructedImage{
			Nx:             output_volume.Nx,
			Ny:             output_volume.Ny,
			SliceThickness: native_slice_thickness,
			Z:              slice_locations,
		}

	}

}
