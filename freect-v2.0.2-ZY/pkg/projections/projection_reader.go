package projections

type ProjectionReader interface {
	ReadProjection() (*Projection, error)
}
