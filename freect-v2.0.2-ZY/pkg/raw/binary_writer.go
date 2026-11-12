package raw

import (
	"encoding/binary"
	"gopkg.in/yaml.v3"
)

var byteOrder = binary.BigEndian

type binaryWriter struct {
	fileStore FileStoreWriter
}

func NewBinaryWriter(f FileStoreWriter) *binaryWriter {
	return &binaryWriter{
		fileStore: f,
	}
}

func (b *binaryWriter) Write(r RawReader) error {
	w_meta, err := b.fileStore.Create("metadata.yaml")
	if err != nil {
		return err
	}

	geom, err := r.ReadGeometry()
	err = yaml.NewEncoder(w_meta).Encode(geom)
	if err != nil {
		return err
	}

	w_proj, err := b.fileStore.Create("projections.dat")
	if err != nil {
		return err
	}

	p, err := r.ReadProjection()
	for p != nil {
		if err != nil {
			return err
		}

		err := binary.Write(w_proj, byteOrder, p.Data)
		if err != nil {
			return err
		}

		p, err = r.ReadProjection()
	}

	return nil
}
