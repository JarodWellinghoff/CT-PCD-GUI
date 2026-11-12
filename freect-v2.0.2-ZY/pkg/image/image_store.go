package image

import (
	"io"
	"io/fs"
)

// ImageStore abstracts something that holds multiple files. E.g., a
// local directory, dropbox, object store, PACS, etc.  It is intended
// to be very similar to the freect/pkg/raw package's FileStore type.
//
// TODO: Give some consideration to consolidating ImageStore with
// FileStore. It only makes sense to keep them separate if we're going
// to do something image-specific/raw specific.
type ImageStore interface {
	ImageStoreReader
	ImageStoreWriter
}

type ImageStoreWriter interface {
	String() string
	Create(name string) (io.Writer, error)
}

type ImageStoreReader interface {
	Stat(name string) (fs.FileInfo, error)
	Open(name string) (fs.File, error)
	ListFiles() ([]string, error)
}
