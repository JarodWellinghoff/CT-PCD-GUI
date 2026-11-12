package raw

import (
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

// Verify at compile that we fulfill the FileStore interface (includes read and write)
var _ FileStore = new(localDirectoryStore)

// local directory implements a raw FileStore for local directories
type localDirectoryStore struct {
	Path string
}

func NewLocalDirectoryStore(path string) *localDirectoryStore {
	return &localDirectoryStore{
		Path: path,
	}
}

func (l *localDirectoryStore) String() string {
	return "local directory store for " + l.Path
}

func (l *localDirectoryStore) Create(name string) (io.Writer, error) {
	fullpath := filepath.Join(l.Path, name)

	f, err := os.Create(fullpath)
	if err != nil {
		return nil, err
	}

	return f, nil
}

func (l *localDirectoryStore) Open(name string) (fs.File, error) {
	fullpath := filepath.Join(l.Path, name)

	f, err := os.Open(fullpath)
	if err != nil {
		return nil, err
	}

	return f, nil
}

func (l *localDirectoryStore) ListFiles() ([]string, error) {
	files, err := os.ReadDir(l.Path)
	if err != nil {
		return nil, err
	}

	file_list := make([]string, len(files))
	for i, f := range files {
		//file_list[i] = filepath.Join(l.Path, f.Name())
		file_list[i] = f.Name()
	}

	return file_list, nil
}

func (l *localDirectoryStore) Stat(name string) (fs.FileInfo, error) {
	return os.Stat(filepath.Join(l.Path, name))
}
