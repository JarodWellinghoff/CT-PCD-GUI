package filter

import (
	"gitlab.com/freect/freect/pkg/projections"
	"testing"
	//"gitlab.com/freect/freect/pkg/geometry"
	"gitlab.com/freect/freect/pkg/raw"
	"gitlab.com/freect/freect/pkg/recon/rebin"
	"time"
)

// These are not REAL tests yet.  Just quick ways to generate data
// that can then be verified externally.  I will need to think about
// how we want to do longer-term, standalone testing.

//var test_data = "/home/john/Data/sample_case/"
var test_data = "/Users/jhoffman/Data/ldct/sample_case/"

func TestFilter(t *testing.T) {

	t.Log("Loading projections... (may take a moment)")

	d := raw.NewLocalDirectoryStore(test_data)
	r, err := raw.NewBattelleReader(d)
	if err != nil {
		t.Fatal("BattelleReader failed: ", err)
	}

	projs, geom, err := r.ReadN(1000)
	if err != nil {
		t.Fatal("Failed to read raw data")
	} else {
		t.Logf("Read %d projections", len(projs))
	}

	rebinned, err := rebin.RebinNoFFS(geom, projs)
	if err != nil {
		t.Fatal("Rebin failed: ", err)
	}

	// Generate the filter
	t.Log("Generating filter")
	f := NewNooFilter(geom, 1.0, 1.0)
	err = f.SaveSpatialBinary("filter_test.bin")
	if err != nil {
		t.Fatal("Failed to save: ", err)
	}

	// Save results
	err = f.SaveSpatialJSON("filter_test.json")
	if err != nil {
		t.Fatal("Failed to save: ", err)
	}

	err = f.SaveFourierJSON("filter_test_imag.json")
	if err != nil {
		t.Fatal("Failed to save: ", err)
	}

	// Filter our projection data
	t.Log("Filtering... (may take a moment)")
	start := time.Now()
	filtered_projs, err := Filter(geom, rebinned, f)
	if err != nil {
		t.Fatal("Filter threw an error: ", err)
	}
	t.Logf("Filtered %v projections in %v", len(filtered_projs), time.Since(start))

	// Save results
	err = projections.ProjectionStack(filtered_projs).SaveRowSheet(32)
	if err != nil {
		t.Error("Failed to save row sheet: ", err)
	}

	err = projections.ProjectionStack(filtered_projs).SaveRaw("filter_stack.bin")
	if err != nil {
		t.Error("Failed to save row sheet: ", err)
	}

}
