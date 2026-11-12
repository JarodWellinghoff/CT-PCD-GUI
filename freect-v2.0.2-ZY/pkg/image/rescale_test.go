package image

import (
	"testing"
)

func GenerateTestImage() *ReconstructedImage {
	tmp := NewReconstructedImage(2, 2)
	for i := 0; i < 4; i++ {
		tmp.Data[i] = float32(0.1 * float64(i))
	}
	return tmp
}

func TestRescale(t *testing.T) {

	img := GenerateTestImage()

	t.Log("Original values: ", img.Data)

	img.RescaleToHU(.1)

	t.Log("Rescaled values: ", img.Data)

	if img.Data[0] >= 0 {
		t.Error("Expected first element to be less than zero")
	}

	if img.Data[1] != 0 {
		t.Error("Expected second element equal to zero")
	}

	if img.Data[2] <= 0 {
		t.Error("Expected third element to be greater than zero")
	}

	if img.Data[3] <= 0 {
		t.Error("Expected fourth element to be greater than zero")
	}

}
