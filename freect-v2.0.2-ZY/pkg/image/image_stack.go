package image

import (
	"math"
	"bytes"
	"encoding/binary"
)

type ImageStack []*ReconstructedImage

func (is ImageStack) Read() ([]*ReconstructedImage, error) {
	return is, nil
}

func (is ImageStack) CompressToUint16() (compressed_img []uint16, slope float32, intercept float32) {

	// Determine min max vals for entire stack
	v_max := -math.MaxFloat64
	v_min := math.MaxFloat64

	for _, img := range is {
		tmp_min, tmp_max := img.MinMaxVoxelVal()
		v_max = math.Max(v_max, float64(tmp_max))
		v_min = math.Min(v_min, float64(tmp_min))
	}

	// Compress the images
	compressed_img = make([]uint16, len(is)*is[0].Nx*is[0].Ny)

	slope = float32(v_max - v_min) / float32(math.MaxUint16)
	intercept = float32(v_min)

	for slice_idx, img := range is {
		offset := slice_idx * img.Nx * img.Ny
		for i, v := range img.Data {
			compressed_img[i+offset] = uint16(math.MaxUint16 * (float64(v) - v_min) / (v_max - v_min))
		}
	}

	return

}

// Bytes returns the raw image data as a byte array.  Useful for
// saving to Nifti file format.
func (is ImageStack) Bytes() ([]byte) {

	data := &bytes.Buffer{}

	for _, img := range is {
		binary.Write(data, binary.LittleEndian, img.Data)
	}
	
	return data.Bytes()
}