package raw

import (
	"errors"
)

// Configure where the reconstruction should source its raw data
// Only ONE of these fields can be filled out for a reconstruction
type RawInputConfig struct {
	Battelle string `json:"Batelle" yaml:"Battelle"`
	Binary   string `json:"Binary" yaml:"Binary"`
	Ptr      string `json:"Ptr" yaml:"Ptr"`
	Ctd      string `json:"Ctd" yaml:"Ctd"`
}

func (r *RawInputConfig) EnforceOneChoice() error {
	num_raw := 0

	if r.Battelle != "" {
		num_raw++
	}
	if r.Binary != "" {
		num_raw++
	}
	if r.Ptr != "" {
		num_raw++
	}
	if r.Ctd != "" {
		num_raw++
	}

	if num_raw == 0 {
		return errors.New("no raw data specified.  must specify exactly one (1).")
	}

	if num_raw > 1 {
		return errors.New("multiple raw data inputs specified.  must specify exactly one (1).")
	}

	return nil
}

// GetReader returns a RawReader suitable for reading the raw data
// file(s) specified in the input configuration.  GetReader returns an
// error if RawInputConfig does not have exactly one field set.
// TODO: Extend support beyond LocalDirectoryStore
func (r *RawInputConfig) GetReader() (RawReader, error) {
	err := r.EnforceOneChoice()
	if err != nil {
		return nil, err
	}

	if r.Battelle != "" {
		d := NewLocalDirectoryStore(r.Battelle)
		return NewBattelleReader(d)
	}

	if r.Binary != "" {
		d := NewLocalDirectoryStore(r.Binary)
		return NewBinaryReader(d)
	}

	if r.Ptr != "" {
		return nil, errors.New("no reader currently available for ptr format")
	}

	if r.Ctd != "" {
		return nil, errors.New("no reader currently available for ctd format")
	}

	return nil, errors.New("did not match a suitable reader (you should never see this error message)")
}
