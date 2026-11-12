package raw

import (
	"encoding/binary"
	"fmt"
	//	log "github.com/sirupsen/logrus"
	//"bufio"
	"gitlab.com/freect/freect/pkg/geometry"
	"gitlab.com/freect/freect/pkg/projections"
	"gopkg.in/yaml.v3"
	//"io"
	//"math"
	"errors"
	"time"
)

// Verify at compile that we fulfill the RawReader interface
var _ RawReader = new(binaryReader)

type binaryReader struct {
	fileStore      FileStoreReader
	projectionData []float32
	geom           *geometry.ScanGeometry
	lastRead       int
	rawReader      // embed the rawReader to capture the implementation of Read()
}

func NewBinaryReader(f FileStoreReader) (*binaryReader, error) {

	has_meta := false
	has_projections := false

	fl, err := f.ListFiles()
	if err != nil {
		return nil, err
	}

	fmt.Println(fl)

	for _, f := range fl {
		if f == "metadata.yaml" {
			has_meta = true
		}

		if f == "projections.dat" {
			has_projections = true
		}
	}

	if !has_meta {
		return nil, errors.New("file store for binary data missing metadata.yaml")
	}

	if !has_projections {
		return nil, errors.New("file store for binary data missing projections.dat")
	}

	return &binaryReader{
		lastRead:       -1,
		fileStore:      f,
		geom:           nil,
		projectionData: nil,
	}, nil
}

func (b *binaryReader) ReadGeometry() (*geometry.ScanGeometry, error) {

	// Cache the geometry structure into the reader
	// since we need to reuse it a fair bit for this particular
	// reader
	if b.geom == nil {
		g := &geometry.ScanGeometry{}
		reader, err := b.fileStore.Open("metadata.yaml")
		if err != nil {
			return nil, err
		}

		err = yaml.NewDecoder(reader).Decode(g)
		if err != nil {
			return nil, err
		}

		b.geom = g
	}

	return b.geom, nil
}

// binary.Read is quite slow (seconds for ~4GB of data) compared to
// C-style loading (hundreds of ms for ~4GB of data), however we can
// guarantee type safety, and we get nice byte-order read/write
// support, which would be platform specific in the previous C implementation.
//
// I'm going to leave this as-is for the time being and move onto other more
// useful stuff, but this should be revisited in the future.
//
// I think the only remaining approach would be to use unsafe casting
// of the read-in byte
// slice. https://stackoverflow.com/questions/11924196/convert-between-slices-of-different-types
// Although this is fast, we lose all type-safety guarantees, and the
// byte ordering becomes 100% platform specific.
func (b *binaryReader) loadProjections() error {

	g, err := b.ReadGeometry()
	if err != nil {
		return err
	}

	if b.projectionData == nil {
		start := time.Now()

		num_floats := g.DetectorRows * g.DetectorChannels * g.NumProjectionsTotal
		b.projectionData = make([]float32, num_floats)

		reader, err := b.fileStore.Open("projections.dat")
		if err != nil {
			return err
		}

		if err := binary.Read(reader, byteOrder, b.projectionData); err != nil {
			return err
		}

		fmt.Printf("binary projection data loading took: %v\n", time.Since(start))
	}

	return nil
}

// Attempt to accelerate the read (bufio gains are lost in the
// conversion, AND this uses double the memory).
//func (b *binaryReader) loadProjections2() error {
//
//	g, err := b.ReadGeometry()
//	if err != nil {
//		return err
//	}
//
//	if b.projectionData == nil {
//
//		num_floats := g.DetectorRows * g.DetectorChannels * g.NumProjectionsTotal
//		b.projectionData = make([]float32, num_floats)
//
//		reader, err := b.fileStore.Open("projections.dat")
//		if err != nil {
//			return err
//		}
//
//		tmp := make([]byte, num_floats*8)
//		r := bufio.NewReader(reader)
//
//		start := time.Now()
//		if _, err := io.ReadFull(r, tmp); err != nil {
//			panic(err)
//		}
//
//		fmt.Printf("binary projection data loading (w/o type conversion) took: %v\n", time.Since(start))
//
//		for i, _ := range b.projectionData {
//			offset := i * 8
//			bits := binary.LittleEndian.Uint64(tmp[offset:(offset + 8)])
//			float := math.Float32frombits(bits)
//
//			b.projectionData[i] = float
//		}
//
//	}
//
//	return nil
//
//}

func (b *binaryReader) ReadProjection() (*projections.Projection, error) {

	// Grab the scan geometry
	g, err := b.ReadGeometry()
	if err != nil {
		return nil, err
	}

	// Ensure that we've already loaded the projection data
	if err := b.loadProjections(); err != nil {
		return nil, err
	}

	// Set up the projection read
	curr_proj := b.lastRead + 1

	if curr_proj == g.NumProjectionsTotal {
		return nil, nil
	}

	projection_size := g.DetectorRows * g.DetectorChannels
	offset := projection_size * curr_proj

	p := &projections.Projection{
		Index:       curr_proj,
		Data:        b.projectionData[offset:(offset + projection_size)],
		NumRows:     g.DetectorRows,
		NumChannels: g.DetectorChannels,
		// TODO: table position
		// TODO: tube angle
	}

	b.lastRead = curr_proj

	return p, nil
}

func (b *binaryReader) String() string {
	return "binary format reader, " + b.fileStore.String()
}

func (b *binaryReader) Read(geom_overrides *geometry.ScanGeometry) ([]*projections.Projection, *geometry.ScanGeometry, error) {
	// There are efficiencies that could be made here, but for now we
	// opt for the minimal effort implementation
	geom, err := b.ReadGeometry()
	if err != nil {
		return nil, nil, err
	}

	projs := make([]*projections.Projection, 0, 2048)
	for {
		p, err := b.ReadProjection()
		if err != nil {
			return nil, nil, err
		}

		if p == nil {
			break
		}

		projs = append(projs, p)
	}

	return projs, geom, nil
}
