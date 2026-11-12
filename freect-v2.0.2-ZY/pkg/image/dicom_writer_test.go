package image

import (
	"testing"
	//dicom "gitlab.com/freect/go-dicom"
	"encoding/json"
	"gitlab.com/freect/freect/pkg/raw"
	dicom "gitlab.com/freect/go-dicom"
	"os"
	"time"
)

func RandomImageStack(n int) []*ReconstructedImage {

	imgs := make([]*ReconstructedImage, n)

	loc := 0.0
	increment := 1.0

	for i := range imgs {
		imgs[i] = NewReconstructedImage(128, 128)
		imgs[i].Z = loc + float64(i)*increment
		imgs[i].RandomImageData()
	}

	return imgs
}

func TestImageToDataset(t *testing.T) {
	r := NewReconstructedImage(2, 2)
	r.FieldOfView = 100.0
	r.XOrigin = 10.0
	r.YOrigin = 10.0
	r.Z = 101.01
	r.SliceThickness = 10.0
	r.RescaledToHU = true

	dcm_data, err := ImageToDataset(r, 64)
	if err != nil {
		t.Error(err)
	}

	data, err := json.MarshalIndent(dcm_data, "", "  ")
	if err != nil {
		t.Error(err)
	}

	t.Log(string(data))

	f, err := os.Create("dicom_sample.dcm")
	if err != nil {
		t.Fatal(err)
	}

	defer f.Close()

	err = dicom.Write(f, dcm_data)
	if err != nil {
		t.Fatal(err)
	}
}

func TestDataAsString(t *testing.T) {
	tm := time.Date(2022, time.Month(12), 25, 0, 0, 0, 0, time.UTC)

	res := DateAsString(tm)

	exp := "20221225"
	if res != exp {
		t.Error("Got", res, "but expected", exp)
	}

	tm = time.Date(2022, time.Month(7), 1, 0, 0, 0, 0, time.UTC)

	res = DateAsString(tm)
	exp = "20220701"
	if res != exp {
		t.Error("Got", res, "but expected", exp)
	}
}

func TestDicomWriter(t *testing.T) {

	ld := raw.NewLocalDirectoryStore("./")
	dw := NewDICOMWriter(ld, "")

	ris := RandomImageStack(10)

	err := dw.Write(ImageStack(ris))
	if err != nil {
		t.Error(err)
	}
}
