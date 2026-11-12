package image

import (
	"math"
)

func InRange(val float64, start float64, end float64) bool {

	min := math.Min(start, end)
	max := math.Max(start, end)

	if val < min {
		return false
	} else if val > max {
		return false
	} else {
		return true
	}

}

// GenerateSliceLocationsForNativeBackprojection calculates slice
// locations and pads additional slices onto the beginning and end of
// volume to handle slice thicknessing.
//
// The "pad" argument should be provided in millimeters and generally
// should be equal to one full additional output slice
// thickness. I.e. If 10mm slices are desired in the final image
// stack, pad = 10.
func GenerateSliceLocationsForNativeBackprojection(start, end, slice_spacing, pad float64) []float64 {
	slice_locations := make([]float64, 0)

	var direction float64
	if start == end {
		direction = 1.0
	} else {
		direction = (end - start) / math.Abs(end-start)
	}

	// This pads additional slices onto the beginning and end of our
	// vector. This is not the most efficient approach if
	// slice_spacing >> slice_thickness, however it will be a robust
	// approach to obtain all required slices for proper thicknessing
	start -= direction * pad
	end += direction * pad

	if start == end {
		slice_locations = append(slice_locations, start)
	} else {

		curr_slice := start

		for InRange(curr_slice, start, end) {
			slice_locations = append(slice_locations, curr_slice)
			curr_slice += (direction * slice_spacing)
		}
	}

	return slice_locations
}

// GenerateSliceLocations calculates and returns a vector of slice
// locations based on start, end, and slice spacing.  First element of
// the returned slice is "start" and the last element is "end", or the
// closest calculated value if slice spacing does not evenly divide
// the provided range (where range = end - start).
func GenerateSliceLocations(start float64, end float64, slice_spacing float64) []float64 {
	slice_locations := make([]float64, 0)
	direction := (end - start) / math.Abs(end-start)

	if start == end {
		slice_locations = append(slice_locations, start)
	} else {
		curr_slice := start
		for InRange(curr_slice, start, end) {
			slice_locations = append(slice_locations, curr_slice)
			curr_slice += (direction * slice_spacing)
		}
	}

	return slice_locations
}
