package raw

type RawWriter interface {
	Write(RawReader) error
}
