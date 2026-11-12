package projections

import (
	"encoding/binary"
	"fmt"
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/util"
	"os"
	"sync"
)

type ProjectionStack []*Projection

func (p ProjectionStack) ToArrays() (projections []float32, table_positions []float64, tube_angles []float64) {

	n_rows := p[0].NumRows
	n_channels := p[0].NumChannels
	n_proj := len(p)

	projections = make([]float32, n_rows*n_channels*n_proj)
	table_positions = make([]float64, n_proj)
	tube_angles = make([]float64, n_proj)

	for i, proj := range p {
		table_positions[i] = proj.TablePosition
		tube_angles[i] = proj.TubeAngle

		offset := i * n_rows * n_channels
		for j, v := range proj.Data {
			projections[offset+j] = v
		}
	}

	return
}

func (p ProjectionStack) SaveRaw(file_name string) error {
	f, err := os.Create(file_name)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, proj := range p {
		binary.Write(f, binary.LittleEndian, proj.Data)
	}

	return nil
}

func (p ProjectionStack) RowSheet(row_idx int) []float32 {

	n_channels := p[0].NumChannels
	n_proj := len(p)

	sheet := make([]float32, n_proj*n_channels)

	offset := row_idx * n_channels

	for i, proj := range p {
		for j, val := range proj.Data[offset:(offset + n_channels)] {
			sheet[i*n_channels+j] = val
		}
	}

	return sheet
}

func (p ProjectionStack) SaveRowSheet(row_idx int) error {
	file_name := fmt.Sprintf("row_sheet_%d.bin", row_idx)
	f, err := os.Create(file_name)
	if err != nil {
		return err
	}
	defer f.Close()

	err = binary.Write(f, binary.LittleEndian, p.RowSheet(row_idx))

	if err != nil {
		return err
	}

	return nil
}

// ReverseChannelOrder applies projection.ReverseChannelOrder to all
// projections in the stack.
func (p ProjectionStack) ReverseChannelOrder() {
	// TODO: Convert this to use LaunchParallelN

	n_proj := len(p)
	proj_idxs := make([]int, n_proj)
	for i := 0; i < n_proj; i++ {
		proj_idxs[i] = i
	}

	flipProjection := func(workload []int, wg *sync.WaitGroup) {
		for _, v := range workload {
			p[v].ReverseChannelOrder()
		}

		wg.Done()
	}
	util.LaunchParallel(flipProjection, proj_idxs, util.MAX_PROC)
}

// ReverseRowOrder applies projection.ReverseRowOrder to all
// projections in the stack.
func (p ProjectionStack) ReverseRowOrder() {
	flipRows := func(workload []int, wg *sync.WaitGroup) {
		for _, v := range workload {
			p[v].ReverseRowOrder()
		}
		wg.Done()
	}
	util.LaunchParallelN(flipRows, len(p), util.MAX_PROC)
}

// NegateTubeAngle sets proj.TubeAngle = -proj.TubeAngle.  This can be
// useful when debugging geometry mismatches between datasets.
func (p ProjectionStack) NegateTubeAngle() {
	for _, proj := range p {
		proj.TubeAngle = -proj.TubeAngle
	}
}

func (p ProjectionStack) AdaptiveFiltration(filter_func func(*Projection)) {
	log.Warn("Adaptive filtration not yet implemented!")
}

// CorrectMATLABIndexing applies projection.CorrectMATLABIndexing to
// all projections in the stack.
func (p ProjectionStack) CorrectMATLABIndexing() {
	// Operation is so minimal that we should NOT use threads to try
	// and accelerate
	n_proj := len(p)
	for i := 0; i < n_proj; i++ {
		p[i].CorrectMATLABIndexing()
	}
}
