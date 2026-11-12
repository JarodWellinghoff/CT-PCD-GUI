package recon

import (
	"gitlab.com/freect/freect/pkg/image"
)

// A ReconstructionFunc accepts a reconstruction config and uses it to
// produce reconstructed image data.  Saving of image data must be
// handled separately.
//
// The second argument to a ReconstructionFunc is used to pass in the
// original ReconSpecific pointer used to capture command line overrides.
//
// TODO: Should we narrow the scope of a ReconstructionFunc?
// I.e. instead of taking in the config, perhaps this function should
// take in []*proj.Projection and metadata and produce our
// reconstructed images.  This could promote better separation of
// concerns. Since it's how I'm handling the saving the images.
type ReconstructionFunc func(*ReconConfig, interface{}) ([]*image.ReconstructedImage, error)
