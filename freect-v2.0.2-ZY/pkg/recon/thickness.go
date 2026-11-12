package recon

import (
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/image"
	"math"
	"time"
)

type WindowFunc func(slice_location, sample_location, slice_thickness float64) float64

// Compile time enforcement that we obey the windowfunc signature
var _ WindowFunc = TriangleWindow

func TriangleWindow(slice_location, sample_location, slice_thickness float64) float64 {
	return math.Max(0.0, 1.0-math.Abs(slice_location-sample_location)/slice_thickness)
}

// Thickness resamples a reconstructed volume into a new volume using
// a window function to set slice thickness and location.
func Thickness(output_volume *ReconVolume, image_stack []*image.ReconstructedImage, window WindowFunc) ([]*image.ReconstructedImage, error) {

	// TODO: Alert the user if the incoming image_stack is NOT
	// contiguous, or if slices are thicker than some threshold (1-2mm?)

	// Grab some important metadata from the incoming stack
	nx := image_stack[0].Nx
	ny := image_stack[0].Ny
	fov := image_stack[0].FieldOfView
	x_origin := image_stack[0].XOrigin
	y_origin := image_stack[0].YOrigin

	// Get the locations of the output image stack
	output_slice_locations := image.GenerateSliceLocations(output_volume.StartPosition, output_volume.EndPosition, output_volume.SliceSpacing)
	final_image_stack := make([]*image.ReconstructedImage, len(output_slice_locations))

	// Generate the thickened slices
	start := time.Now()
	log.Info("Thicknessing to final images...")
	for i, v := range output_slice_locations {

		// Create the image and set metadata
		img := image.NewReconstructedImage(output_volume.Nx, output_volume.Ny)

		img.Nx = nx
		img.Ny = ny
		img.FieldOfView = fov
		img.XOrigin = x_origin
		img.YOrigin = y_origin

		img.Z = v
		img.SliceThickness = output_volume.SliceThickness

		// For each input_image in the image_stack, calculate if it
		// will contribute to the current slice, if so, add it and
		// keep track of the weight
		total_weight := 0.0
		for _, input_image := range image_stack {
			weight := window(img.Z, input_image.Z, img.SliceThickness)

			if weight == 0 {
				continue
			}

			for px_idx, px_val := range input_image.Data {
				img.Data[px_idx] += float32(weight * float64(px_val))
			}

			total_weight += weight
		}

		// Normalize using the total_weight
		for px_idx, px_val := range img.Data {
			img.Data[px_idx] = float32(float64(px_val) / total_weight)
		}

		// Save into the final image stack
		final_image_stack[i] = img
	}

	log.Info("Thicknessed images in ", time.Since(start))

	return final_image_stack, nil
}
