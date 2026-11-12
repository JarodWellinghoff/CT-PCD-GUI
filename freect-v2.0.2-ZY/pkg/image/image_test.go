package image

import (
	"testing"
)

func TestInRange(t *testing.T) {

	tf := InRange(0.0, 1.0, 2.0) // NO
	if tf != false {
		t.Error("Expected false but got true")
	}

	tf = InRange(0.0, 2.0, 1.0) // NO
	if tf != false {
		t.Error("Expected false but got true")
	}

	tf = InRange(1.0, 0.0, 2.0) // YES
	if tf != true {
		t.Error("Expected true but got false")
	}

	tf = InRange(1.0, 2.0, 0.0) // YES
	if tf != true {
		t.Error("Expected true but got false")
	}

	tf = InRange(1.0, 1.0, 0.0) // YES
	if tf != true {
		t.Error("Expected true but got false")
	}

	tf = InRange(0.0, 0.0, 1.0) // YES
	if tf != true {
		t.Error("Expected true but got false")
	}

	tf = InRange(0.0, 0.0, 0.0) // YES
	if tf != true {
		t.Error("Expected true but got false")
	}

}

func TestNativeLocations(t *testing.T) {
	// TODO: write an actual test here (just going to use printf debugging for the time being)
	slice_locations := GenerateSliceLocationsForNativeBackprojection(5.0, 15.0, 0.6, 10.0)
	t.Log(slice_locations)
}

func TestSliceLocations(t *testing.T) {
	// Single Slice
	slice_locations := GenerateSliceLocations(1.0, 1.0, 0.6)

	if len(slice_locations) != 1 {
		t.Error("Expected one slice but got ", len(slice_locations))
	}

	if slice_locations[0] != 1.0 {
		t.Error("Expected slice location to be 1.0 but got ", slice_locations[0])
	}

	t.Log("Single slice location:", slice_locations)

	// Multi-slice
	// Increasing direction
	slice_locations = GenerateSliceLocations(1.0, 10.0, 1.0)
	if len(slice_locations) != 10 {
		t.Error("Expected ten slices but got ", len(slice_locations))
	}

	if slice_locations[0] != 1.0 && slice_locations[1] != 2.0 && slice_locations[9] != 10.0 {
		t.Error("Expected slice locations 0, 1, 9 to be 1.0, 2.0, and 10.0 but got ", slice_locations[0], slice_locations[1], slice_locations[9])
	}

	t.Log("Increasing slice locations:", slice_locations)

	// Decreasing direction
	slice_locations = GenerateSliceLocations(10.0, 1.0, 1.0)
	if len(slice_locations) != 10 {
		t.Error("Expected ten slices but got ", len(slice_locations))
	}

	if slice_locations[0] != 10.0 && slice_locations[1] != 9.0 && slice_locations[0] != 1.0 {
		t.Error("Expected slice locations 0, 1, 9 to be 10.0, 9.0, and 1.0 but got ", slice_locations[0], slice_locations[1], slice_locations[9])
	}

	t.Log("Decreasing slice locations:", slice_locations)

}
