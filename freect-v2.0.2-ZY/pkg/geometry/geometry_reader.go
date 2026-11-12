package geometry

type ScanGeometryReader interface {
	ReadGeometry() (*ScanGeometry, error)
}
