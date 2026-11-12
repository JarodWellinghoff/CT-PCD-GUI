package raw

import (
	"gitlab.com/freect/freect/pkg/geometry"
	"gitlab.com/freect/freect/pkg/projections"
)

type RawReader interface {
	Read(geom_overrides *geometry.ScanGeometry) ([]*projections.Projection, *geometry.ScanGeometry, error)
	String() string
	projections.ProjectionReader
	geometry.ScanGeometryReader
}

type rawReader struct {
	projections.ProjectionReader
	geometry.ScanGeometryReader
}

// Compose a RawReader from separate ProjectionReader and
// ScanGeometryReaders.  Generally speaking, this is probably not
// necessary for most applications.
func NewRawReader(p projections.ProjectionReader, g geometry.ScanGeometryReader) RawReader {
	return &rawReader{
		ProjectionReader:   p,
		ScanGeometryReader: g,
	}
}

func (r *rawReader) String() string {
	return "raw reader"
}

func (r *rawReader) Read(geom_overrides *geometry.ScanGeometry) ([]*projections.Projection, *geometry.ScanGeometry, error) {
	g, err := r.ScanGeometryReader.ReadGeometry()
	if err != nil {
		return nil, nil, err
	}

	projs := make([]*projections.Projection, 0, 2048)
	for {
		p, err := r.ProjectionReader.ReadProjection()
		if err != nil {
			return nil, nil, err
		}

		if p == nil {
			break
		}

		projs = append(projs, p)
	}

	return projs, g, nil
}
