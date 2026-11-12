package recon

import (
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/projections"
)

func Preprocess(preprocessing_config *PreprocessingConfig, projs []*projections.Projection) ([]*projections.Projection, error) {

	if preprocessing_config.CorrectMATLABIndexing {
		log.Info("preprocessing: correcting matlab indexing")
		projections.ProjectionStack(projs).CorrectMATLABIndexing()
	}

	if preprocessing_config.FlipDetectorChannels {
		log.Info("preprocessing: flipping detector channels")
		projections.ProjectionStack(projs).ReverseChannelOrder()
	}

	if preprocessing_config.FlipDetectorRows {
		log.Info("preprocessing: flipping detector rows")
		projections.ProjectionStack(projs).ReverseRowOrder()
	}

	if preprocessing_config.NegateTubeAngle {
		log.Info("preprocessing: negating tube angle")
		projections.ProjectionStack(projs).NegateTubeAngle()
	}

	if preprocessing_config.AdapativeFiltration > 0 {
		log.Info("preprocessing: adaptive filtration kachelreiss")
		filter_func := AdaptiveFiltrationKachelreiss
		projections.ProjectionStack(projs).AdaptiveFiltration(filter_func)
	}

	return projs, nil
}
