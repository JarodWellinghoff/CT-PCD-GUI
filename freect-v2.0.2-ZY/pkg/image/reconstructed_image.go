package image

import (
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
)

//type ImageStack []*ReconstructedImage

type ReconstructedImage struct {
	Nx             int
	Ny             int
	FieldOfView    float64
	XOrigin        float64
	YOrigin        float64
	Z              float64
	SliceThickness float64
	Data           []float32
	RescaledToHU   bool
}

func NewReconstructedImage(nx, ny int) *ReconstructedImage {
	return &ReconstructedImage{
		Nx:   nx,
		Ny:   ny,
		Data: make([]float32, nx*ny),
	}
}

func (r *ReconstructedImage) SavePNG(filename string, black_level float64, white_level float64) error {

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	img := image.NewGray(image.Rect(0, 0, r.Nx, r.Ny))

	for i := 0; i < r.Nx; i++ {
		for j := 0; j < r.Ny; j++ {

			idx := i + r.Nx*j
			val := float64(r.Data[idx])

			var c color.Gray
			if val <= black_level {
				c.Y = uint8(0)
			} else if val >= white_level {
				c.Y = uint8(255)
			} else {
				c.Y = uint8(256 * (val - black_level) / (white_level - black_level))
			}

			img.Set(i, j, c)
		}
	}

	err = png.Encode(f, img)
	if err != nil {
		return err
	}

	return nil
}

func (r *ReconstructedImage) SaveDebugPNG(filename string) error {
	// Determine the min/max brightness of the image
	min := math.MaxFloat32
	max := -math.MaxFloat32
	for _, val := range r.Data {
		val_64 := float64(val)
		if val_64 < min {
			min = val_64
		}

		if val_64 > max {
			max = val_64
		}
	}

	return r.SavePNG(filename, min, max)
}

func (r *ReconstructedImage) SaveBinary(filename string) error {

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	err = binary.Write(f, binary.LittleEndian, r.Data)
	if err != nil {
		return err
	}

	return nil
}

func (r *ReconstructedImage) RescaleToHU(mu_water float64) error {
	if r.RescaledToHU {
		return errors.New("Image already rescaled HU")
	}

	if mu_water <= 0.0 {
		return fmt.Errorf("Invalid water attenuation (%f) for HU rescale (must be greater than 0.0)", mu_water)
	}

	for i, mu := range r.Data {
		r.Data[i] = float32(math.Round(1000.0 * (float64(mu) - mu_water) / mu_water))
	}

	r.RescaledToHU = true
	return nil
}

func (r *ReconstructedImage) RandomImageData() {
	for i := range r.Data {
		r.Data[i] = float32(rand.Intn(2048) - 1024)
	}
}

func (r *ReconstructedImage) MinMaxVoxelVal() (min, max float32) {

	tmp_max := -math.MaxFloat64
	tmp_min := math.MaxFloat64

	for _, v := range r.Data {
		tmp_max = math.Max(tmp_max, float64(v))
		tmp_min = math.Min(tmp_min, float64(v))
	}

	max = float32(tmp_max)
	min = float32(tmp_min)

	return
}

func (r *ReconstructedImage) CompressToUint16() (compressed_img []uint16, slope float32, intercept float32) {

	v_min, v_max := r.MinMaxVoxelVal()

	compressed_img = make([]uint16, len(r.Data))

	slope = (v_max-v_min) / float32(math.MaxUint16)
	intercept = v_min

	for i, v := range r.Data {
		compressed_img[i] = uint16(math.MaxUint16 * (v - v_min) / (v_max - v_min))
	}

	return
}

// Brainstorm: Generate a preallocated stack of image data from a
// recon.ReconVolume?  NOTE: good idea, but only could be used in the
// final image stack (i.e. not suitable for the native thickness
// backprojection)
