package raw

import (
	"io"
	"io/fs"
)

// FileStore is an abstraction for something that holds multiple
// files.  E.g., a local directory, dropbox directory, etc.  If this
// issue https://github.com/golang/go/issues/45757 gets addressed and
// added to Go's standard library, this will be replaced by that.
type FileStore interface {
	FileStoreReader
	FileStoreWriter
}

type FileStoreWriter interface {
	Create(name string) (io.Writer, error)
}

type FileStoreReader interface {
	String() string // Return descriptive string about the file store
	Stat(name string) (fs.FileInfo, error)
	Open(name string) (fs.File, error) // Open a file for reading
	ListFiles() ([]string, error)      // List files in directory (note: should only be passed back to Open for use)
}
