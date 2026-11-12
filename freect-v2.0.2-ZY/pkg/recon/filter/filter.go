package filter

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"github.com/mjibson/go-dsp/dsputils"
	"github.com/mjibson/go-dsp/fft"
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/geometry"
	"gitlab.com/freect/freect/pkg/projections"
	"gitlab.com/freect/freect/pkg/util"
	"math"
	"os"
	"sync"
)

type filter struct {
	NumProjectionChannels int
	NumFilterChannels     int
	spatial               []float64
	fourier               []complex128
}

func (f *filter) SaveSpatialBinary(filename string) error {
	fh, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer fh.Close()
	binary.Write(fh, binary.LittleEndian, f.spatial)
	return nil
}

func (f *filter) SaveSpatialJSON(filename string) error {
	fh, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer fh.Close()

	data, err := json.MarshalIndent(f.spatial, "", "  ")
	if err != nil {
		return err
	}

	fh.Write(data)
	return nil
}

func (f *filter) SaveFourierBinary(filename string) error {
	tmp := make([]float64, len(f.fourier))

	for i, v := range f.fourier {
		tmp[i] = real(v)
	}

	fh, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer fh.Close()
	binary.Write(fh, binary.LittleEndian, tmp)
	return nil
}

func (f *filter) SaveFourierJSON(filename string) error {
	tmp := make([]float64, len(f.fourier))

	for i, v := range f.fourier {
		//fmt.Printf("c: %v; r: %v\n", v, real(v))
		tmp[i] = real(v)
	}

	data, err := json.MarshalIndent(tmp, "", "  ")
	if err != nil {
		return err
	}

	fh, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer fh.Close()

	fh.Write(data)

	return nil
}

func NewNooFilter(scan_geom *geometry.ScanGeometry, c float64, a float64) *filter {

	// TODO: Make upsampling factor a tuneable parameter?
	// Generate the filter at 2*N_channels
	N := 4 * scan_geom.DetectorChannels // 1776 // Will depend on the upsampling factor
	pi := math.Pi
	ds := scan_geom.DetectorTransverseSpacing * scan_geom.DistSourceToIsocenter / scan_geom.DistSourceToDetector

	filter_spatial := make([]float64, N)

	r := func(t float64) float64 {
		if t == 0 {
			return 0.5
		}
		return math.Sin(t)/t + (math.Cos(t)-1.0)/(t*t)
	}

	for i := -N / 2; i < N/2; i++ {
		filter_spatial[i+N/2] = (c * c / (2.0 * ds)) * (a*r(c*pi*float64(i)) +
			(((1.0 - a) / 2.0) * r(pi*c*float64(i)+pi)) +
			(((1.0 - a) / 2.0) * r(pi*c*float64(i)-pi)))
	}

	// After generation, zero pad out to 4*N_Channels to avoid cyclic convolution effects
	// Further zero pad out to nearest power of 2 to leverage Cooley-Tukey algorithm
	N_pad := dsputils.NextPowerOf2(N)
	filter_spatial = dsputils.ZeroPadF(filter_spatial, N_pad) // only on the right of the signal

	return &filter{
		NumProjectionChannels: N,
		NumFilterChannels:     N_pad,
		spatial:               filter_spatial,
		fourier:               fft.FFTReal(filter_spatial), // just the fft from float64 -> complex128
	}

}

// Filter projections
func multiplyElements(v1 []complex128, v2 []complex128) ([]complex128, error) {
	if len(v1) != len(v2) {
		return nil, fmt.Errorf("multiplyElements failed: len(v1)=%v != len(v2)=%v", len(v1), len(v2))
	}

	result := make([]complex128, len(v1))
	for i := range result {
		result[i] = v1[i] * v2[i]
	}

	return result, nil
}

func Filter(scan_geom *geometry.ScanGeometry, projs []*projections.Projection, filter *filter) ([]*projections.Projection, error) {

	N_pad := filter.NumFilterChannels

	if projs[0].Rebinned == false {
		log.Warn("Projections have not been rebinned prior to filtering")
	}

	fft.SetWorkerPoolSize(1)
	filterProj := func(workload []int, wg *sync.WaitGroup) {
		for _, w := range workload {

			proj := projs[w]

			util.LogUpdate(w,len(projs), 10, "filtering projections ")
			
			//fmt.Printf("filtering projection %v/%v\n", w+1, len(projs))

			// Extract the current row, process it, then store the filtered projection back in place of the original
			for j := 0; j < proj.NumRows; j++ {

				// Extract row, convert to float64, and pad
				offset := j * proj.NumChannels

				curr_row := proj.Data[offset:(offset + proj.NumChannels)]
				curr_row_f64 := make([]float64, len(curr_row))
				for i, v := range curr_row {
					curr_row_f64[i] = float64(v)
				}

				curr_row_pad := dsputils.ZeroPadF(curr_row_f64, N_pad)

				multiplied, err := multiplyElements(filter.fourier, fft.FFTReal(curr_row_pad))
				if err != nil {
					return
				}

				result := fft.IFFT(multiplied)

				// Copy data back into our projection
				// 1/N scaling is done by the IFFT function in go-dsp/fft (i.e. we don't have to explicitly do it as in cufft)
				for k := 0; k < proj.NumChannels; k++ {
					proj.Data[offset+k] = float32(real(result[k+proj.NumChannels])) // "Zero frequency" of our filter occurs at proj.NumChannels/2, hence the offset
				}

			}
			proj.Filtered = true
		}
		wg.Done()
	}
	util.LaunchParallelN(filterProj, len(projs), util.MAX_PROC)

	// Again, operating on the data "in place" could easily create
	// confusion.  We should reeeeally consider if this is the best
	// approach...
	return projs, nil
}
