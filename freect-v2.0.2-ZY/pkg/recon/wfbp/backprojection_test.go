package wfbp

import (
	"fmt"
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/raw"
	"gitlab.com/freect/freect/pkg/recon"
	"gitlab.com/freect/freect/pkg/recon/filter"
	"gitlab.com/freect/freect/pkg/recon/rebin"
	//"image"
	"gitlab.com/freect/freect/pkg/projections"
	"testing"
	"time"
)

var test_data = "/Users/jhoffman/Data/ldct/sample_case/"

func TestBackprojection(t *testing.T) {

	t.Log("Loading projections... (may take a moment)")

	d := raw.NewLocalDirectoryStore(test_data)
	r, err := raw.NewBattelleReader(d)
	if err != nil {
		t.Fatal("BattelleReader failed: ", err)
	}

	projs, geom, err := r.ReadN(4000)
	if err != nil {
		t.Fatal("Failed to read raw data")
	} else {
		t.Logf("Read %d projections", len(projs))
	}

	rebinned, err := rebin.RebinNoFFS(geom, projs)
	if err != nil {
		t.Fatal("Rebin failed: ", err)
	}

	// TEST
	log.Info("Flipping channel direction after rebinning....")
	projections.ProjectionStack(rebinned).ReverseChannelOrder()

	// Generate the filter
	t.Log("Generating filter")
	f := filter.NewNooFilter(geom, 0.5, 0.5)

	// Filter our projection data
	t.Log("Filtering... (may take a moment)")
	start := time.Now()
	filtered_projs, err := filter.Filter(geom, rebinned, f)
	if err != nil {
		t.Fatal("Filter threw an error: ", err)
	}
	t.Logf("Filtered %v projections in %v", len(filtered_projs), time.Since(start))

	// Backprojection
	config := recon.NewReconConfig(DefaultWFBPConfig())

	slice_location := (filtered_projs[0].TablePosition+filtered_projs[len(filtered_projs)-1].TablePosition)/2.0 - 10.0

	n := 512
	vol := &recon.ReconVolume{
		XOrigin:        0.0,
		YOrigin:        50.0,
		FieldOfView:    450.0,
		StartPosition:  slice_location,
		EndPosition:    slice_location,
		SliceThickness: 2.0,
		Nx:             n,
		Ny:             n,
	}

	config.ReconVolume = vol
	config.ScanGeometry = geom

	t.Log("Backprojecting... (may take a moment)")
	start = time.Now()
	images, err := BackprojectNative(config, filtered_projs)
	if err != nil {
		log.Fatal("backprojection failed: ", err)
	}
	t.Logf("Backprojected %v voxels in %v", n*n, time.Since(start))

	for i, img := range images {
		filename_root := fmt.Sprintf("recon_%v", i)
		err = img.SaveDebugPNG(filename_root + ".png")
		if err != nil {
			t.Error("PNG save failed for file ", filename_root, ": ", err)
		}

		err = img.SaveBinary(filename_root + ".bin")
		if err != nil {
			t.Error("Binary save failed for file ", filename_root, ": ", err)
		}
	}

}
