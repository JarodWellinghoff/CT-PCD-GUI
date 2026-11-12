package nifti

import (
	"crypto/rand"
	"os"
	"testing"
	"encoding/binary"
	"bytes"
	"fmt"
	mathrand "math/rand"
)

func circleSlice(nx, ny int, radius float64,  val int16) []byte {

	x_off := (float64(nx) - 1.0)/2.0
	y_off := (float64(ny) - 1.0)/2.0

	slice := make([]int16, nx*ny)
	for i := 0; i < nx*ny; i++ {

		row := i/nx
		col := i%nx

		x := (float64(col) - x_off) // dx assumed to be 1.0
		y := (float64(row) - y_off) // dx assumed to be 1.0

		// Want radius to be nx/4
		if x*x + y*y < radius*radius {
			slice[i] = val
		}
		
	}

	b := &bytes.Buffer{}
	binary.Write(b, binary.LittleEndian, slice)
	return b.Bytes()

}

func randomSlice(nx, ny int, bytes_per_vox int) []byte {
	buff := make([]byte, nx*ny*bytes_per_vox)
	
	_, err := rand.Read(buff)
	if err != nil {
		fmt.Println("crypto/rand.Read failed?")
	}

	return buff
}


func TestNifti(t *testing.T) {

	nx := 128
	ny := 128
	nz := 64

	img := NewNifti1Image(128, 128, 64, CodeSignedShort)
	//slice_data := make([]byte, nx*ny*2) // using unsigned chars

	for i := 0; i < nz; i++ {

	
		//t.Log(slice_data[0:10])
		v :=  int16(i*10)
		t.Log("Setting slice ", i, "with val", v)
		err := img.SetSliceData(i, circleSlice(nx, ny, 0.25*float64(nx) + 10.0*(mathrand.Float64()-0.5), v))
		//err := img.SetSliceData(i, randomSlice(nx, ny , 2))
		if err != nil {
			t.Error(err)
		}
	}

	f, _ := os.Create("/Users/jhoffman/Desktop/nifti_test.nii.gz")
	defer f.Close()

	err := img.Write(f)
	if err != nil {
		t.Error(err)
	}

}
