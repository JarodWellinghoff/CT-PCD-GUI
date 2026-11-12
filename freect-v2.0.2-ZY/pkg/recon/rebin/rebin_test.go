package rebin

import (
	"fmt"
	"gitlab.com/freect/freect/pkg/projections"
	"gitlab.com/freect/freect/pkg/raw"
	"gopkg.in/yaml.v3"
	"testing"
)

func TestRebinInterp(t *testing.T) {

	projs := make([]*projections.Projection, 2)

	projs[0] = &projections.Projection{
		Index:       0,
		Data:        []float32{0.0, 1.0, 2.0, 3.0},
		NumRows:     2,
		NumChannels: 2,
	}

	projs[1] = &projections.Projection{
		Index:       1,
		Data:        []float32{4.0, 5.0, 6.0, 7.0},
		NumRows:     2,
		NumChannels: 2,
	}

	// Check for correct interpolated values
	v1 := RebinInterp(projs, 0, 0.5, 0)
	if v1 != 0.5 {
		t.Errorf("Expected 0.5 but got %0.9f", v1)
	} else {
		t.Log("Row 0, Projection 0, Channel 0.5: ", v1)
	}

	v2 := RebinInterp(projs, 1.0, 0.5, 0)
	if v2 != 4.5 {
		t.Errorf("Expected 4.5 but got %0.9f", v2)
	} else {
		t.Log("Row 0, Projection 1, Channel 0.5: ", v2)
	}

	v3 := RebinInterp(projs, 0, 0.5, 1)
	if v3 != 2.5 {
		t.Errorf("Expected 2.5 but got %0.9f", v3)
	} else {
		t.Log("Row 1, Projection 1, Channel 0.5: ", v3)
	}

	v4 := RebinInterp(projs, 1.0, 0.5, 1)
	if v4 != 6.5 {
		t.Errorf("Expected 6.5 but got %0.9f", v4)
	} else {
		t.Log("Row 1, Projection 1, Channel 0.5: ", v4)
	}

	v5 := RebinInterp(projs, 0.5, 0, 0)
	if v5 != 2.0 {
		t.Errorf("Expected 2.0 but got %0.9f", v5)
	} else {
		t.Log("Row 0, Projection 0.5, Channel 0: ", v5)
	}

	// Check that we don't panic even w/ out of bounds read
	v6 := RebinInterp(projs, 10, 10, 10)
	if v6 != 7.0 {
		t.Errorf("Expected 7.0 but got %0.9f", v6)
	} else {
		t.Log("Row 10, Projection 10.0, Channel 10.0 (clamped): ", v6)
	}

	v7 := RebinInterp(projs, -1, -1, -1)
	if v7 != 0.0 {
		t.Errorf("Expected 0.0 but got %0.9f", v7)
	} else {
		t.Log("Row -1, Projection -1.0, Channel -1.0 (clamped): ", v7)
	}

}

// Correct way to test this is to load a full study and then rebin it
//var test_data = "/Users/jhoffman/Data/sample_case/" // work machine
var test_data = "/Users/jhoffman/Data/ldct/sample_case/" // mac

func TestRebinNoFFS(t *testing.T) {

	d := raw.NewLocalDirectoryStore(test_data)
	r, err := raw.NewBattelleReader(d)
	if err != nil {
		t.Fatal("BattelleReader failed: ", err)
	}
	projs, geom, err := r.Read()
	if err != nil {
		t.Fatal("Failed to read raw data")
	} else {
		t.Logf("Read %d projections", len(projs))
	}

	data, err := yaml.Marshal(geom)
	fmt.Println("Current geometry:")
	fmt.Println("==================================================")
	fmt.Println(string(data))

	//err = projections.ProjectionStack(projs).SaveRaw("raw.bin")
	//if err != nil {
	//	t.Error(err)
	//}

	t.Logf(`
n_rows: %d
n_chan: %d
`, projs[0].NumRows, projs[0].NumChannels)

	err = projections.ProjectionStack(projs).SaveRowSheet(16)
	if err != nil {
		t.Error("Did not save row sheet: ", err)
	}

	//err = projections.ProjectionStack(projs).SaveRowSheet(32)
	//if err != nil {
	//	t.Error("Did not save sheet: ", err)
	//}
	//t.Fatalf("early exit: (%d, %d)", geom.DetectorChannels, geom.NumProjectionsTotal)

	if geom.FlyingFocalSpotMode != "FFSNONE" {
		t.Fatal("Expected FFSNONE but got ", geom.FlyingFocalSpotMode)
	}

	rebinned, err := RebinNoFFS(geom, projs)
	if err != nil {
		t.Fatal("Rebin failed: ", err)
	}

	err = projections.ProjectionStack(rebinned).SaveRaw("rebinned.bin")
	if err != nil {
		t.Error(err)
	}

	t.Logf(`
n_rows: %d
n_chan: %d
`, rebinned[0].NumRows, rebinned[0].NumChannels)

	err = projections.ProjectionStack(rebinned).SaveRowSheet(32)
	if err != nil {
		t.Error("Did not save sheet: ", err)
	}

}
